#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <array>

namespace py = pybind11;
constexpr int MAX_ACTIONS = 262144;
constexpr int MAX_RANKS = 32;
struct Shared {
  std::atomic<int> state[MAX_ACTIONS]; // 0 pending, 1 ready, 2 GPU complete
  std::atomic<int> arrived[MAX_RANKS];
  std::atomic<int> epoch;
  std::atomic<int> error;
  std::atomic<int> ready_head[MAX_RANKS], done_head[MAX_RANKS];
  int next_ready[MAX_ACTIONS], next_done[MAX_ACTIONS];
  std::atomic<int> completed_epoch;
  std::atomic<int> batch_ready[MAX_RANKS];
};
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Action {
  int src, dst;
  int64_t src_offset, dst_offset, scratch, bytes;
  std::vector<int> deps, resources;
};
struct Plan { std::vector<Action> actions; std::vector<int> order; std::vector<std::vector<int>> stages; int resources; };
struct Skeleton {
  int root_degree, max_count;
  std::vector<std::vector<int>> degrees, roles;
  std::vector<std::vector<int>> arcs; // src switch, role, dst switch, role, stage
  std::vector<std::pair<std::pair<int,int>,int>> cross_counts;
};
struct Flow { int src, mask; std::vector<int64_t> tokens; std::vector<int> dst;
  std::vector<std::array<int,3>> arcs; std::vector<int> lexical_destinations; int stages; };
struct CompactPlan {
  Plan dispatch, combine;
  std::vector<Flow> flows;
  std::vector<std::vector<int64_t>> d_offset,c_offset;
  std::vector<int64_t> capacities;
  std::vector<std::vector<int64_t>> pack_src,pack_dst,expert_global,expert_storage,reduce_src,reduce_dst,expert_combine;
  int world,hidden,tokens,topk;
  int64_t useful=0, candidates=0;
  double grouping_us=0,selection_us=0,layout_us=0;
};
struct CompactPlanner {
  struct Pattern {
    std::array<int,4> canon{},counts{};
    int key=0,min_count=100;
  };
  int world, switches;
  std::vector<Pattern> patterns;
  std::vector<unsigned char> pattern_ready;
  bool planning_started=false;
  double gpu_bw, node_bw, sys_bw;
  std::map<int,std::vector<Skeleton>> catalog;
  CompactPlanner(int w,double gpu,double node,double sys):world(w),switches((w+3)/4),gpu_bw(gpu),node_bw(node),sys_bw(sys){TORCH_CHECK(w>0&&w<=16);patterns.resize(w*(1<<w));pattern_ready.resize(w*(1<<w),0);}
  void add(int key,py::list rows) {
    TORCH_CHECK(!planning_started,"Catalog must be built before preparing routes");
    auto& dst=catalog[key];
    for(auto row:rows) {
      auto r=py::cast<py::tuple>(row);
      dst.push_back({r[0].cast<int>(),r[1].cast<int>(),r[2].cast<std::vector<std::vector<int>>>(),
                     r[3].cast<std::vector<std::vector<int>>>(),r[4].cast<std::vector<std::vector<int>>>()});
      auto& s=dst.back();
      for(auto&a:s.arcs)if(a[0]!=a[2]){
        std::pair<int,int> pair={a[0],a[2]};auto it=std::find_if(s.cross_counts.begin(),s.cross_counts.end(),[&](auto&x){return x.first==pair;});
        if(it==s.cross_counts.end())s.cross_counts.push_back({pair,1});else it->second++;
      }
    }
  }
  CompactPlan prepare(torch::Tensor ids,int experts,int hidden,int mode,int materialize_rank=-1) {
    auto clock_start=std::chrono::steady_clock::now();planning_started=true;
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type()==torch::kInt64 && ids.is_contiguous());
    TORCH_CHECK(world>0 && world<=16 && ids.dim()==3 && ids.size(0)==world,"Expected up to 16 ranks and [world,tokens,topk] ids");
    TORCH_CHECK(experts>0 && experts%world==0 && hidden>0 && mode>=0 && mode<=3,"Invalid expert placement, hidden size, or strategy");
    int tokens=ids.size(1),topk=ids.size(2);const int64_t* id=ids.data_ptr<int64_t>();
    CompactPlan p;p.world=world;p.hidden=hidden;p.tokens=tokens;p.topk=topk;
    std::vector<std::vector<Flow>> per(world);
    std::vector<int> lexical_ranks(world),lexical_rank(world);std::iota(lexical_ranks.begin(),lexical_ranks.end(),0);
    std::sort(lexical_ranks.begin(),lexical_ranks.end(),[](int a,int b){return "g"+std::to_string(a)<"g"+std::to_string(b);});
    for(int i=0;i<world;i++)lexical_rank[lexical_ranks[i]]=i;
    for(int src=0;src<world;src++) {
      std::map<int,std::vector<int64_t>> groups;
      for(int t=0;t<tokens;t++) {
        int mask=0;for(int k=0;k<topk;k++){int64_t e=id[(src*tokens+t)*topk+k];TORCH_CHECK(e>=-1 && e<experts,"Expert ID out of bounds");if(e>=0)mask|=1<<(e/(experts/world));}
        mask&=~(1<<src);if(mask)groups[mask].push_back(t);
      }
      for(auto& [mask,ts]:groups){Flow f;f.src=src;f.mask=mask;f.tokens=ts;
        for(int r=0;r<world;r++)if(mask&(1<<r))f.dst.push_back(r);
        for(int r:f.dst)f.lexical_destinations.push_back(lexical_rank[r]);
        f.stages=0;for(int d=f.dst.size();d;d>>=1)f.stages++;
        per[src].push_back(std::move(f));}
      // Python layout ties compare destination names in rank order.
      std::stable_sort(per[src].begin(),per[src].end(),[](const Flow&a,const Flow&b){
        if(a.tokens.size()!=b.tokens.size())return a.tokens.size()>b.tokens.size();
        return a.lexical_destinations<b.lexical_destinations;});
    }
    for(int i=0;;i++){bool any=false;for(auto& fs:per)if(i<(int)fs.size()){p.flows.push_back(std::move(fs[i]));any=true;}if(!any)break;}
    std::stable_sort(p.flows.begin(),p.flows.end(),[](auto&a,auto&b){return a.tokens.size()>b.tokens.size();});
    auto clock_grouped=std::chrono::steady_clock::now();
    p.grouping_us=std::chrono::duration<double,std::micro>(clock_grouped-clock_start).count();
    int edges=world*2+switches*switches;
    auto up=[](int r){return r*2;};auto down=[](int r){return r*2+1;};
    auto edge=[&](int a,int b){return world*2+a*switches+b;};
    auto bandwidth=[&](int e){if(e<world*2)return gpu_bw;int a=(e-world*2)/switches,b=(e-world*2)%switches;return a/2==b/2?node_bw:sys_bw;};
    auto resources=[&](int a,int b){std::vector<int>x={up(a),down(b)};if(a/4!=b/4)x.push_back(edge(a/4,b/4));return x;};
    std::vector<const Pattern*> bindings(p.flows.size());
    double theta=1e-9,total=0;double minbw=std::min({gpu_bw,node_bw,sys_bw})*1000;
    for(size_t i=0;i<p.flows.size();i++){
      auto& f=p.flows[i];total+=f.tokens.size()*hidden*2*f.dst.size();
      int pattern_index=f.src*(1<<world)+f.mask;
      auto& pattern=patterns[pattern_index];
      if(!pattern_ready[pattern_index]){
        for(int d:f.dst)pattern.counts[d/4]++;
        auto& counts=pattern.counts;auto& canon=pattern.canon;
        canon[0]=f.src/4;int ci=1;for(int sw=0;sw<switches;sw++)if(sw!=f.src/4)canon[ci++]=sw;
        std::sort(canon.begin()+1,canon.begin()+switches,[&](int a,int b){
          if(counts[a]!=counts[b])return counts[a]>counts[b];
          return (a-f.src/4+switches)%switches<(b-f.src/4+switches)%switches;});
        int mul=1;for(int c=0;c<switches;c++){pattern.key+=counts[canon[c]]*mul;mul*=5;}
        auto it=catalog.find(pattern.key);
        if(it!=catalog.end())for(auto&sk:it->second)pattern.min_count=std::min(pattern.min_count,sk.max_count);
        pattern_ready[pattern_index]=1;
      }
      bindings[i]=&pattern;
      if(mode==1||mode==2){
        if(pattern.min_count==100)throw std::runtime_error("Missing native catalog pattern "+std::to_string(pattern.key));
        theta=std::max(theta,pattern.min_count*f.tokens.size()*hidden*2/minbw);}

    }
    p.useful=total;
    // Match the reference model's directed edge count including schema NICs.
    double lambda=2*std::log(world*2+switches*(switches-1)+switches*4)+3;
    theta=std::max(theta,total/(world*minbw*lambda));
    std::vector<double> loads(edges,0);
    for(int round=0;round<=16;round++){
      std::fill(loads.begin(),loads.end(),0);
      std::map<int64_t,std::vector<double>> increment_cache;
      for(size_t i=0;i<p.flows.size();i++){
        auto& f=p.flows[i];f.arcs.clear();double vol=f.tokens.size()*hidden*2;
        if(mode==0||mode==3){for(int d:f.dst)f.arcs.push_back({f.src,d,-1});}
        else {
          const auto& canon=bindings[i]->canon;auto& variants=catalog.at(bindings[i]->key);
          double zero_loads[64]={};const double*current=mode==2?zero_loads:loads.data();
          int sorted[4][4]={},sorted_count[4]={};
          for(int d:f.dst){int c=std::find(canon.begin(),canon.begin()+switches,d/4)-canon.begin();sorted[c][sorted_count[c]++]=d;}
          for(int c=0;c<switches;c++)std::sort(sorted[c],sorted[c]+sorted_count[c],[&](int a,int b){
            if(current[up(a)]!=current[up(b)])return current[up(a)]<current[up(b)];
            if(current[down(a)]!=current[down(b)])return current[down(a)]<current[down(b)];
            return lexical_rank[a]<lexical_rank[b];});
          double best_score=INFINITY;int best=0;
          double localtheta=theta;
          if(mode==2){int count=bindings[i]->min_count;localtheta=std::max(count*vol/minbw,vol*f.dst.size()/(world*minbw*lambda));}
          double inv=1.0/(2.0*localtheta);
          std::vector<double>*increments=nullptr;
          if(mode==1){auto& table=increment_cache[int64_t(vol)];if(table.empty()){
            table.resize(edges*17);for(int e=0;e<edges;e++)for(int count=1;count<=16;count++){
              double inc=count*vol/(bandwidth(e)*1000);table[e*17+count]=std::expm1(std::min(60.,inc*inv));}}
            increments=&table;}
          double cached_score[64][17];bool score_ready[64][17]={};
          double current_exp[64];bool exp_ready[64]={};
          for(int v=0;v<(int)variants.size();v++){
            p.candidates++;auto& s=variants[v];double score=0;
            auto add_score=[&](int e,int count){if(count){double inc=count*vol/(bandwidth(e)*1000);
              if(!score_ready[e][count]){
                if(!exp_ready[e]){current_exp[e]=std::exp(std::min(60.,current[e]*inv));exp_ready[e]=true;}
                cached_score[e][count]=current_exp[e]*(increments?(*increments)[e*17+count]:std::expm1(std::min(60.,inc*inv)));score_ready[e][count]=true;}
              score+=cached_score[e][count];}};
            add_score(up(f.src),s.root_degree);
            for(int c=0;c<switches;c++)for(int pos=0;pos<(int)s.degrees[c].size();pos++)
              add_score(up(sorted[c][pos]),s.degrees[c][pos]);
            for(auto& [pair,count]:s.cross_counts)add_score(edge(canon[pair.first],canon[pair.second]),count);
            if(score<best_score){best_score=score;best=v;}
          }
          auto&s=variants[best];int roles[4][17]={};roles[0][0]=f.src;
          for(int c=0;c<switches;c++)for(int j=0;j<(int)s.roles[c].size();j++)roles[c][s.roles[c][j]+1]=sorted[c][j];
          for(auto&a:s.arcs)f.arcs.push_back({roles[a[0]][a[1]+1],roles[a[2]][a[3]+1],a[4]});
        }
        int counts[64]={};
        for(auto&a:f.arcs){counts[up(a[0])]++;counts[down(a[1])]++;if(a[0]/4!=a[1]/4)counts[edge(a[0]/4,a[1]/4)]++;}
        for(int e=0;e<edges;e++)if(counts[e])loads[e]+=counts[e]*vol/(bandwidth(e)*1000);
      }
      if(mode!=1||*std::max_element(loads.begin(),loads.end())<=lambda*theta)break;
      theta*=2;
    }
    auto clock_selected=std::chrono::steady_clock::now();
    p.selection_us=std::chrono::duration<double,std::micro>(clock_selected-clock_grouped).count();
    int count=p.flows.size();p.capacities.resize(world,0);p.d_offset.assign(world,std::vector<int64_t>(count,-1));p.c_offset=p.d_offset;
    p.pack_src.resize(world);p.pack_dst.resize(world);p.expert_global.resize(world);p.expert_storage.resize(world);p.reduce_src.resize(world);p.reduce_dst.resize(world);p.expert_combine.resize(world);
    int64_t rowbytes=int64_t(hidden)*2;
    int64_t alignment=std::lcm(int64_t(256),rowbytes);
    auto alloc=[&](int r,int64_t bytes){auto& x=p.capacities[r];x=(x+alignment-1)/alignment*alignment;int64_t out=x;x+=bytes;return out;};
    for(int r=0;r<world;r++)if(materialize_rank<0||r==materialize_rank)for(int t=0;t<tokens;t++) {p.expert_global[r].push_back(r*tokens+t);p.expert_storage[r].push_back(-t-1);p.expert_combine[r].push_back(-1);}
    std::vector<int> dispatch_depth,combine_depth;
    std::vector<double> priorities;
    for(int fi=0;fi<count;fi++){
      auto& f=p.flows[fi];int64_t bytes=f.tokens.size()*rowbytes;
      std::vector<int> nodes=f.dst;nodes.push_back(f.src);
      for(int r:nodes){p.d_offset[r][fi]=alloc(r,bytes);p.c_offset[r][fi]=alloc(r,bytes);}
      if(materialize_rank<0||f.src==materialize_rank)for(int j=0;j<(int)f.tokens.size();j++){
        p.pack_src[f.src].push_back(f.tokens[j]);p.pack_dst[f.src].push_back(p.d_offset[f.src][fi]/rowbytes+j);
        p.reduce_src[f.src].push_back(p.c_offset[f.src][fi]/rowbytes+j);p.reduce_dst[f.src].push_back(f.tokens[j]);
      }
      for(int r:f.dst)if(materialize_rank<0||r==materialize_rank)for(int j=0;j<(int)f.tokens.size();j++){
        p.expert_global[r].push_back(f.src*tokens+f.tokens[j]);
        p.expert_storage[r].push_back(p.d_offset[r][fi]/rowbytes+j);
        p.expert_combine[r].push_back(p.c_offset[r][fi]/rowbytes+j);
      }
      int base=p.dispatch.actions.size();
      if(materialize_rank>=0)continue;
      for(int ai=0;ai<(int)f.arcs.size();ai++){
        auto&a=f.arcs[ai];int src=a[0],dst=a[1];
        std::vector<int> dd,cd;for(int j=0;j<(int)f.arcs.size();j++){
          if(f.arcs[j][1]==src)dd.push_back(base+j);if(f.arcs[j][0]==dst)cd.push_back(base+j);}
        p.dispatch.actions.push_back({src,dst,p.d_offset[src][fi],p.d_offset[dst][fi],0,bytes,dd,resources(src,dst)});
        p.combine.actions.push_back({dst,src,p.c_offset[dst][fi],p.c_offset[src][fi],alloc(src,bytes),bytes,cd,resources(dst,src)});
        dispatch_depth.push_back(a[2]>=0?f.stages-a[2]:0);
        int depth=1,node=src;
        while(node!=f.src){auto parent=std::find_if(f.arcs.begin(),f.arcs.end(),[&](auto&arc){return arc[1]==node;});TORCH_CHECK(parent!=f.arcs.end(),"Disconnected tree");node=(*parent)[0];depth++;}
        combine_depth.push_back(depth);
        double priority=0;for(int e:resources(src,dst))priority=std::max(priority,loads[e]);priorities.push_back(priority);
      }
    }
    for(Plan* plan:{&p.dispatch,&p.combine}){
      plan->resources=edges;plan->order.resize(plan->actions.size());std::iota(plan->order.begin(),plan->order.end(),0);
      // Reference priority: remaining binomial stage for dispatch, tree depth
      // for combine, then the actual forward path load for both phases.
      const auto& depth=plan==&p.dispatch?dispatch_depth:combine_depth;
      std::stable_sort(plan->order.begin(),plan->order.end(),[&](int i,int j){if(depth[i]!=depth[j])return depth[i]>depth[j];
        double a=priorities[i],b=priorities[j];
        if(a!=b)return a>b;return plan->actions[i].bytes>plan->actions[j].bytes;});
    }
    if(mode==3){
      // Repository CollWay-like: direct, endpoint-exclusive greedy stages;
      // combine executes the exact reversed dispatch stages.
      auto pending=p.dispatch.order;
      std::vector<int> action_flow(p.dispatch.actions.size());int ai=0;
      for(int fi=0;fi<count;fi++)for(auto&a:p.flows[fi].arcs)action_flow[ai++]=fi;
      std::sort(pending.begin(),pending.end(),[&](int i,int j){
        if(priorities[i]!=priorities[j])return priorities[i]>priorities[j];
        if(action_flow[i]!=action_flow[j])return action_flow[i]<action_flow[j];
        return lexical_rank[p.dispatch.actions[i].dst]<lexical_rank[p.dispatch.actions[j].dst];});
      while(!pending.empty()){
        std::vector<bool> send(world,false),receive(world,false);std::vector<int> stage,next;
        for(int i:pending){auto&a=p.dispatch.actions[i];if(send[a.src]||receive[a.dst]){next.push_back(i);continue;}
          send[a.src]=receive[a.dst]=true;stage.push_back(i);}
        p.dispatch.stages.push_back(std::move(stage));pending=std::move(next);
      }
      p.combine.stages=p.dispatch.stages;std::reverse(p.combine.stages.begin(),p.combine.stages.end());
    }
    if(p.dispatch.actions.size()>MAX_ACTIONS)throw std::runtime_error("Compact action capacity exceeded");
    p.layout_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-clock_selected).count();
    return p;
  }
};
static Plan make_plan(py::list rows, std::vector<int> order, int resources) {
  Plan p; p.order = std::move(order); p.resources = resources;
  for (auto row : rows) {
    auto r = py::cast<py::tuple>(row);
    p.actions.push_back({r[0].cast<int>(), r[1].cast<int>(), r[2].cast<int64_t>(),
      r[3].cast<int64_t>(), r[4].cast<int64_t>(), r[5].cast<int64_t>(),
      r[6].cast<std::vector<int>>(), r[7].cast<std::vector<int>>()});
  }
  if (p.actions.size() > MAX_ACTIONS) throw std::runtime_error("IPC action capacity exceeded");
  return p;
}
template<typename T> __global__ void add_kernel(T* out, const T* x, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += int64_t(blockDim.x)*gridDim.x)
    out[i] = T(float(out[i]) + float(x[i]));
}
template<> __global__ void add_kernel<__nv_bfloat16>(__nv_bfloat16* out, const __nv_bfloat16* x, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += int64_t(blockDim.x)*gridDim.x)
    out[i] = __float2bfloat16_rn(__bfloat162float(out[i]) + __bfloat162float(x[i]));
}
static void create_shared(const std::string& path) {
  int fd = open(path.c_str(), O_CREAT|O_EXCL|O_RDWR, 0600);
  if (fd < 0) throw std::runtime_error("Cannot create IPC control file");
  if (ftruncate(fd, sizeof(Shared))) { close(fd); throw std::runtime_error("Cannot size IPC control file"); }
  close(fd);
}
// Batched peer reads: one block per output row, many independent flows per
// launch. Combine owns one accumulator per block and sums all children.
// Vectorized aligned row transport; keep FP32 child-add order unchanged.
__global__ void batch_peer_kernel(const int64_t* rows,const int64_t* pointers,int count,bool combine){
  int row=blockIdx.x;if(row>=count)return;const int64_t*d=rows+row*36;
  auto*out=(__nv_bfloat16*)(pointers[d[0]]+d[1]);int n=d[2],children=d[3];
  if(n%8==0){
    for(int i=threadIdx.x*8;i<n;i+=blockDim.x*8){
      if(!combine){
        auto*src=(const __nv_bfloat16*)(pointers[d[4]]+d[5]);
        *reinterpret_cast<uint4*>(out+i)=*reinterpret_cast<const uint4*>(src+i);
      }else{
        uint4 initial=*reinterpret_cast<const uint4*>(out+i);
        auto* initial_values=reinterpret_cast<__nv_bfloat16*>(&initial);
        float values[8];
        #pragma unroll
        for(int j=0;j<8;j++)values[j]=__bfloat162float(initial_values[j]);
        for(int c=0;c<children;c++){
          auto*src=(const __nv_bfloat16*)(pointers[d[4+2*c]]+d[5+2*c]);
          uint4 input=*reinterpret_cast<const uint4*>(src+i);
          auto* input_values=reinterpret_cast<__nv_bfloat16*>(&input);
          #pragma unroll
          for(int j=0;j<8;j++)values[j]+=__bfloat162float(input_values[j]);
        }
        uint4 output;auto* output_values=reinterpret_cast<__nv_bfloat16*>(&output);
        #pragma unroll
        for(int j=0;j<8;j++)output_values[j]=__float2bfloat16_rn(values[j]);
        *reinterpret_cast<uint4*>(out+i)=output;
      }
    }
  }else{
    if(!combine){auto*src=(const __nv_bfloat16*)(pointers[d[4]]+d[5]);
      for(int i=threadIdx.x;i<n;i+=blockDim.x)out[i]=src[i];return;}
    for(int i=threadIdx.x;i<n;i+=blockDim.x){float value=__bfloat162float(out[i]);
      for(int c=0;c<children;c++){auto*src=(const __nv_bfloat16*)(pointers[d[4+2*c]]+d[5+2*c]);value+=__bfloat162float(src[i]);}
      out[i]=__float2bfloat16_rn(value);}
  }
}
static std::vector<torch::Tensor> batch_descriptors(const Plan&plan,int rank,bool combine){
  std::vector<int> levels(plan.actions.size(),0);int max_level=0;
  for(size_t iter=0;iter<plan.actions.size();iter++){
    bool changed=false;for(int i:plan.order)for(int d:plan.actions[i].deps)if(levels[i]<=levels[d]){levels[i]=levels[d]+1;changed=true;}
    if(!changed)break;TORCH_CHECK(iter<32,"Invalid dependency depth");}
  for(int l:levels)max_level=std::max(max_level,l);
  // Combine actions writing the same accumulator must be in one batch after
  // every child is ready; promote siblings until all dependencies are safe.
  if(combine)for(int iter=0;iter<32;iter++){
    bool changed=false;std::map<std::pair<int,int64_t>,int> group;
    for(int i:plan.order)group[{plan.actions[i].dst,plan.actions[i].dst_offset}]=std::max(group[{plan.actions[i].dst,plan.actions[i].dst_offset}],levels[i]);
    for(int i:plan.order){int l=group[{plan.actions[i].dst,plan.actions[i].dst_offset}];for(int d:plan.actions[i].deps)l=std::max(l,levels[d]+1);
      if(l>levels[i]){levels[i]=l;changed=true;}max_level=std::max(max_level,levels[i]);}
    if(!changed)break;TORCH_CHECK(iter<31,"Combine level convergence failed");}
  std::vector<torch::Tensor> tensors;
  for(int level=0;level<=max_level;level++){
    std::map<std::pair<int64_t,int64_t>,std::vector<int>> groups;
    for(int i:plan.order)if(plan.actions[i].dst==rank&&levels[i]==level){auto&a=plan.actions[i];groups[{a.dst_offset,a.bytes}].push_back(i);}
    std::vector<int64_t> values;
    for(auto&[key,actions]:groups){TORCH_CHECK(actions.size()<=16);for(int64_t byte=0;byte<key.second;byte+=4096){
      size_t base=values.size();values.resize(base+36,0);values[base]=rank;values[base+1]=key.first+byte;
      values[base+2]=std::min(int64_t(4096),key.second-byte)/2;values[base+3]=actions.size();
      for(int j=0;j<(int)actions.size();j++){auto&a=plan.actions[actions[j]];values[base+4+2*j]=a.src;values[base+5+2*j]=a.src_offset+byte;}}}
    auto cpu=torch::empty({int64_t(values.size()/36),36},torch::TensorOptions().dtype(torch::kInt64));
    if(!values.empty())memcpy(cpu.data_ptr(),values.data(),values.size()*sizeof(int64_t));
    tensors.push_back(cpu.to(torch::kCUDA));
  }
  return tensors;
}
static std::vector<torch::Tensor> compact_batch_descriptors(const CompactPlan&p,int rank,bool combine){
  std::vector<std::vector<int64_t>> levels(16);int max_level=0;
  for(size_t fi=0;fi<p.flows.size();fi++){
    auto&f=p.flows[fi];int parent[16],depth[16]={},height[16]={};std::fill(parent,parent+16,-1);
    int children[16][16]={},child_count[16]={};for(auto&a:f.arcs){parent[a[1]]=a[0];children[a[0]][child_count[a[0]]++]=a[1];}
    for(int pass=0;pass<16;pass++){bool changed=false;for(auto&a:f.arcs){
      int d=depth[a[0]]+1,h=height[a[1]]+1;
      if(d>depth[a[1]]){depth[a[1]]=d;changed=true;}if(h>height[a[0]]){height[a[0]]=h;changed=true;}}
      if(!changed)break;}
    for(auto&a:f.arcs)max_level=std::max(max_level,combine?height[a[0]]-1:depth[a[1]]-1);
    if((!combine&&parent[rank]<0)||(combine&&child_count[rank]==0))continue;
    int level=combine?height[rank]-1:depth[rank]-1;auto&values=levels[level];
    int64_t bytes=int64_t(f.tokens.size())*p.hidden*2;
    for(int64_t byte=0;byte<bytes;byte+=4096){size_t base=values.size();values.resize(base+36,0);
      values[base]=rank;values[base+1]=(combine?p.c_offset[rank][fi]:p.d_offset[rank][fi])+byte;
      values[base+2]=std::min(int64_t(4096),bytes-byte)/2;values[base+3]=combine?child_count[rank]:1;
      if(combine)for(int j=0;j<(int)child_count[rank];j++){int src=children[rank][j];values[base+4+j*2]=src;values[base+5+j*2]=p.c_offset[src][fi]+byte;}
      else{values[base+4]=parent[rank];values[base+5]=p.d_offset[parent[rank]][fi]+byte;}
    }
  }
  std::vector<torch::Tensor> tensors;for(int level=0;level<=max_level;level++){
    auto&values=levels[level];auto cpu=torch::empty({int64_t(values.size()/36),36},torch::TensorOptions().dtype(torch::kInt64));
    if(!values.empty())memcpy(cpu.data_ptr(),values.data(),values.size()*sizeof(int64_t));tensors.push_back(cpu.to(torch::kCUDA));
  }
  return tensors;
}
class Executor {
  int rank, world, fd, epoch = 0,batch_epoch=0;
  Shared* shared;
  std::vector<char*> bases;
  std::vector<cudaStream_t> streams;
  cudaStream_t reduction;
  torch::Tensor device_pointers;
public:
  Executor(std::string path, int r, int w):rank(r),world(w) {
    if (world > MAX_RANKS) throw std::runtime_error("Too many IPC ranks");
    fd = open(path.c_str(), O_RDWR);
    if (fd < 0) throw std::runtime_error("Cannot open IPC control file");
    shared = (Shared*)mmap(nullptr,sizeof(Shared),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if (shared == MAP_FAILED) throw std::runtime_error("Cannot mmap IPC control file");
    for (int i=0;i<world;i++) { cudaStream_t s; check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking)); streams.push_back(s); }
    check(cudaStreamCreateWithFlags(&reduction,cudaStreamNonBlocking));
  }
  void unmap() {
    check(cudaDeviceSynchronize());
    for (int i=0;i<(int)bases.size();i++) if(i!=rank) check(cudaIpcCloseMemHandle(bases[i]));
    bases.clear();
  }
  void map(std::vector<std::string> handles, uint64_t pointer) {
    bases.resize(world);
    for(int i=0;i<world;i++) {
      if(i==rank) { bases[i]=(char*)pointer; continue; }
      if(handles[i].size()!=sizeof(cudaIpcMemHandle_t)) throw std::runtime_error("Invalid IPC handle");
      cudaIpcMemHandle_t h; memcpy(&h,handles[i].data(),sizeof(h));
      check(cudaIpcOpenMemHandle((void**)&bases[i],h,cudaIpcMemLazyEnablePeerAccess));
    }
    std::vector<int64_t> pointers;for(auto*p:bases)pointers.push_back(int64_t(p));
    device_pointers=torch::tensor(pointers,torch::TensorOptions().dtype(torch::kInt64)).to(torch::kCUDA);
  }
  void run_batch(torch::Tensor rows,bool combine){
    if(rows.size(0)){batch_peer_kernel<<<rows.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(rows.data_ptr<int64_t>(),device_pointers.data_ptr<int64_t>(),rows.size(0),combine);check(cudaGetLastError());}
  }
  void batch_fence_impl(){
    check(cudaDeviceSynchronize());++batch_epoch;
    shared->batch_ready[rank].store(batch_epoch,std::memory_order_release);
    auto start=std::chrono::steady_clock::now();
    for(int r=0;r<world;r++)while(shared->batch_ready[r].load(std::memory_order_acquire)<batch_epoch){
      if(shared->error.load(std::memory_order_acquire))throw std::runtime_error("Batch peer failed");
      if(std::chrono::steady_clock::now()-start>std::chrono::seconds(60)){shared->error.store(1,std::memory_order_release);throw std::runtime_error("Batch readiness timeout");}
    }
  }
  void batch_fence(){py::gil_scoped_release release;batch_fence_impl();}
  void run_batches(const std::vector<torch::Tensor>& levels,bool combine){
    py::gil_scoped_release release;
    for(const auto&rows:levels){batch_fence_impl();run_batch(rows,combine);}
    batch_fence_impl();
  }
  void close_all() {
    unmap();
    for(auto s:streams) check(cudaStreamDestroy(s));
    streams.clear();check(cudaStreamDestroy(reduction));
    munmap(shared,sizeof(Shared));::close(fd);
  }
  void run(const Plan& plan, bool combine, bool exclusive, bool bf16) {
    py::gil_scoped_release release;
    ++epoch;
    auto start=std::chrono::steady_clock::now();
    auto guard=[&]() {
      if(shared->error.load(std::memory_order_acquire)) throw std::runtime_error("IPC peer execution failed");
      if(std::chrono::steady_clock::now()-start > std::chrono::seconds(60)) {
        shared->error.store(1,std::memory_order_release);throw std::runtime_error("IPC executor timeout");
      }
    };
    if(rank==0) {
      for(size_t i=0;i<plan.actions.size();i++) shared->state[i].store(0,std::memory_order_relaxed);
      shared->epoch.store(epoch,std::memory_order_release);
    } else while(shared->epoch.load(std::memory_order_acquire)!=epoch) guard();
    std::vector<int> status(plan.actions.size(),0), resource(plan.resources,0);
    std::vector<cudaEvent_t> events(plan.actions.size(),nullptr);
    size_t complete=0,current_stage=0;
    try {
      while(complete<plan.actions.size()) {
        if(rank==0) {
          for(int i:plan.order) {
            auto& a=plan.actions[i];
            if(status[i]==1 && shared->state[i].load(std::memory_order_acquire)==2) {
              status[i]=2;if(exclusive)for(int e:a.resources)resource[e]=0;
            }
          }
          if(!plan.stages.empty())while(current_stage<plan.stages.size()){
            bool done=true;for(int i:plan.stages[current_stage])if(status[i]!=2){done=false;break;}
            if(!done)break;current_stage++;
          }
          const auto& eligible=plan.stages.empty()?plan.order:plan.stages[std::min(current_stage,plan.stages.size()-1)];
          for(int i:eligible){auto&a=plan.actions[i];if(status[i])continue;
            bool ready=true;for(int d:a.deps)if(status[d]!=2){ready=false;break;}
            if(exclusive)for(int e:a.resources)if(resource[e]){ready=false;break;}
            if(!ready)continue;status[i]=1;if(exclusive)for(int e:a.resources)resource[e]=1;
            shared->state[i].store(1,std::memory_order_release);
          }
        }
        complete=0;
        for(int i:plan.order){const auto&a=plan.actions[i];int state=shared->state[i].load(std::memory_order_acquire);
          if(state==2){complete++;continue;}if(a.dst!=rank||state!=1)continue;
          if(!events[i]){
            auto stream=streams[a.src];char*dst=bases[rank]+(combine?a.scratch:a.dst_offset);
            check(cudaMemcpyAsync(dst,bases[a.src]+a.src_offset,a.bytes,cudaMemcpyDefault,stream));
            check(cudaEventCreateWithFlags(&events[i],cudaEventDisableTiming));check(cudaEventRecord(events[i],stream));
            if(combine){check(cudaStreamWaitEvent(reduction,events[i],0));int64_t n=a.bytes/(bf16?2:4);int blocks=std::min<int64_t>(256,(n+255)/256);
              if(blocks){if(bf16)add_kernel<<<blocks,256,0,reduction>>>((__nv_bfloat16*)(bases[rank]+a.dst_offset),(__nv_bfloat16*)dst,n);
                else add_kernel<<<blocks,256,0,reduction>>>((float*)(bases[rank]+a.dst_offset),(float*)dst,n);check(cudaGetLastError());}
              check(cudaEventRecord(events[i],reduction));}
          }else{auto e=cudaEventQuery(events[i]);if(e==cudaSuccess)shared->state[i].store(2,std::memory_order_release);else if(e!=cudaErrorNotReady)check(e);}
        }
        guard();
      }
      for(auto e:events)if(e)check(cudaEventDestroy(e));
      shared->arrived[rank].store(epoch,std::memory_order_release);
      // No rank may reset the shared states until every rank has observed the
      // previous call's completion. The next Python readiness barrier also
      // protects reuse of payload allocations across calls.
      for(int r=0;r<world;r++)while(shared->arrived[r].load(std::memory_order_acquire)<epoch)guard();
    } catch(...) {shared->error.store(1,std::memory_order_release);throw;}
  }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("allocate",[](int64_t bytes){
    void* p;check(cudaMalloc(&p,bytes));int device;check(cudaGetDevice(&device));
    return torch::from_blob(p,{bytes},[](void* x){cudaFree(x);},
                           torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA,device));
  });
  m.def("create_shared",&create_shared);
  m.def("handle",[](torch::Tensor t){cudaIpcMemHandle_t h;check(cudaIpcGetMemHandle(&h,t.data_ptr()));return py::bytes((char*)&h,sizeof(h));});
  py::class_<Plan>(m,"Plan",py::module_local()).def("transfers",[](const Plan&p){
    py::list rows;for(auto&a:p.actions)rows.append(py::make_tuple(a.src,a.dst,a.src_offset,a.dst_offset,a.bytes));return rows;})
    .def_property_readonly("action_count",[](const Plan&p){return p.actions.size();})
    .def_readonly("stages",&Plan::stages);
  m.def("make_plan",&make_plan);
  m.def("batch_descriptors",&batch_descriptors);
  m.def("compact_batch_descriptors",&compact_batch_descriptors);
  py::class_<CompactPlan>(m,"CompactPlan",py::module_local())
    .def_readonly("dispatch",&CompactPlan::dispatch).def_readonly("combine",&CompactPlan::combine)
    .def_readonly("capacities",&CompactPlan::capacities).def_readonly("pack_src",&CompactPlan::pack_src)
    .def_readonly("pack_dst",&CompactPlan::pack_dst).def_readonly("expert_global",&CompactPlan::expert_global)
    .def_readonly("expert_storage",&CompactPlan::expert_storage).def_readonly("expert_combine",&CompactPlan::expert_combine)
    .def_readonly("reduce_src",&CompactPlan::reduce_src).def_readonly("reduce_dst",&CompactPlan::reduce_dst)
    .def_readonly("useful",&CompactPlan::useful).def_readonly("candidates",&CompactPlan::candidates)
    .def_readonly("grouping_us",&CompactPlan::grouping_us).def_readonly("selection_us",&CompactPlan::selection_us).def_readonly("layout_us",&CompactPlan::layout_us)
    .def("indices",[](const CompactPlan&p,int r){TORCH_CHECK(r>=0&&r<p.world);return py::make_tuple(p.pack_src[r],p.pack_dst[r],p.expert_global[r],p.expert_storage[r],p.expert_combine[r],p.reduce_src[r],p.reduce_dst[r]);})
    .def("metrics",[](const CompactPlan&p){
      py::dict out;std::vector<int64_t> injection(p.world,0),sent(p.world,0),received(p.world,0),group_sizes;
      std::map<std::string,int64_t> cuts;int64_t logical=0;
      for(auto&f:p.flows){int64_t bytes=int64_t(f.tokens.size())*p.hidden*2;group_sizes.push_back(f.tokens.size());
        for(auto&a:f.arcs){logical+=bytes;sent[a[0]]+=bytes;received[a[1]]+=bytes;if(a[0]==f.src)injection[f.src]+=bytes;
          cuts["gpu"+std::to_string(a[0])+"-up"]+=bytes;cuts["gpu"+std::to_string(a[1])+"-down"]+=bytes;
          if(a[0]/4!=a[1]/4)cuts["switch"+std::to_string(a[0]/4)+"-"+std::to_string(a[1]/4)]+=bytes;}}
      out["logical_dispatch_bytes"]=logical;out["source_injection_bytes"]=injection;
      out["rank_sent_bytes"]=sent;out["rank_received_bytes"]=received;out["group_tokens"]=group_sizes;
      size_t actions=0;for(auto&f:p.flows)actions+=f.arcs.size();
      out["modeled_directed_cut_bytes"]=cuts;out["flows"]=p.flows.size();out["dispatch_actions"]=actions;
      out["combine_actions"]=actions;return out;
    })
    .def("peer_indices",[](const CompactPlan&p,int rank,bool combine){
      TORCH_CHECK(rank>=0&&rank<p.world);const Plan&plan=combine?p.combine:p.dispatch;
      std::vector<std::vector<int64_t>> send(p.world),recv(p.world);int64_t rb=p.hidden*2;
      for(auto&a:plan.actions){if(a.src==rank)for(int64_t j=0;j<a.bytes/rb;j++)send[a.dst].push_back(a.src_offset/rb+j);
        if(a.dst==rank)for(int64_t j=0;j<a.bytes/rb;j++)recv[a.src].push_back(a.dst_offset/rb+j);}
      std::vector<int64_t> si,ri,sc,rc;for(int r=0;r<p.world;r++){si.insert(si.end(),send[r].begin(),send[r].end());ri.insert(ri.end(),recv[r].begin(),recv[r].end());sc.push_back(send[r].size());rc.push_back(recv[r].size());}
      return py::make_tuple(si,ri,sc,rc);
    })
    .def("trees",[](const CompactPlan&p){py::list out;for(auto&f:p.flows)out.append(py::make_tuple(f.src,f.mask,f.tokens,f.arcs));return out;});
  py::class_<CompactPlanner>(m,"CompactPlanner",py::module_local()).def(py::init<int,double,double,double>())
    .def("add",&CompactPlanner::add).def("prepare",&CompactPlanner::prepare,py::arg("ids"),py::arg("experts"),py::arg("hidden"),py::arg("mode"),py::arg("materialize_rank")=-1);
  py::class_<Executor>(m,"Executor",py::module_local()).def(py::init<std::string,int,int>()).def("map",&Executor::map)
    .def("unmap",&Executor::unmap).def("run",&Executor::run).def("run_batch",&Executor::run_batch).def("batch_fence",&Executor::batch_fence).def("run_batches",&Executor::run_batches).def("close",&Executor::close_all);
}
