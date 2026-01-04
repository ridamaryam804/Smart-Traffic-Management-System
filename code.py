# app_final.py
# Upload-only, stateful Smart City graph tool for location-name distance matrices.
# Includes: BFS, DFS, Dijkstra, A*, Greedy Best-First (GBFS), Brute-Force Shortest Path (small),
# step-by-step trace, graph view, comparison table, dynamic traffic, vehicle scheduling, TSP (NN & exact BnB).

import math, time, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart City — Final (Upload + Pick Matrix, Greedy & Brute Force)", layout="wide")
random.seed(42)

# ===================== Helpers / Data =====================
@dataclass
class Step:
    iteration: int
    current: Optional[str]
    frontier: List[str]
    visited: List[str]
    distances: Dict[str, float]
    parent: Dict[str, Optional[str]]

def reconstruct_path(parent, start, goal):
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        if cur == start: break
        cur = parent.get(cur)
    path.reverse()
    return path if path and path[0]==start and path[-1]==goal else []

def classical_mds(D: np.ndarray, dim=2):
    n = D.shape[0]
    if n < 2: return np.zeros((n, dim))
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * J @ (D**2) @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    pos = np.clip(vals[:dim], 0, None)
    if np.all(pos == 0): return np.zeros((n, dim))
    L = np.diag(np.sqrt(pos))
    return vecs[:, :dim] @ L

def build_graph_from_matrix(dist_df: pd.DataFrame, directed: bool):
    """Return (G, coords, nodes)"""
    if not directed:
        dist_df = (dist_df + dist_df.T) / 2.0
        np.fill_diagonal(dist_df.values, 0)
    nodes = [str(x) for x in dist_df.index]
    G = nx.DiGraph() if directed else nx.Graph()
    for n in nodes:
        G.add_node(n, label=n)
    for u in nodes:
        for v in nodes:
            if u == v: continue
            w = float(dist_df.loc[u, v])
            if w > 0 and not math.isnan(w):
                G.add_edge(u, v, weight=w)
    D = dist_df.values.astype(float)
    coords2d = classical_mds(D, dim=2)
    coords = {n:(float(coords2d[i,0]), float(coords2d[i,1])) for i,n in enumerate(nodes)} \
             if coords2d.size else {n:(i,0.0) for i,n in enumerate(nodes)}
    return G, coords, nodes

# ===================== Algorithms =====================
def bfs(G, start, goal):
    from collections import deque
    q, visited = deque([start]), {start}
    parent={start:None}; steps=[]; it=0
    while q:
        u=q.popleft(); it+=1
        steps.append(Step(it,u,list(q),sorted(visited),{n: math.inf for n in G.nodes},parent.copy()))
        if u==goal: return reconstruct_path(parent,start,goal), steps, it
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v); parent[v]=u; q.append(v)
    return [], steps, it

def dfs(G,start,goal):
    stack=[start]; visited=set(); parent={start:None}; steps=[]; it=0
    while stack:
        u=stack.pop()
        if u in visited: continue
        visited.add(u); it+=1
        steps.append(Step(it,u,list(stack),sorted(visited),{n: math.inf for n in G.nodes},parent.copy()))
        if u==goal: return reconstruct_path(parent,start,goal), steps, it
        for v in reversed(list(G.neighbors(u))):
            if v not in visited:
                if v not in parent: parent[v]=u
                stack.append(v)
    return [], steps, it

def dijkstra(G,start,goal):
    import heapq
    dist={n: math.inf for n in G.nodes}; dist[start]=0.0
    parent={start:None}; pq=[(0.0,start)]; visited=set(); steps=[]; it=0
    while pq:
        d,u=heapq.heappop(pq)
        if u in visited: continue
        visited.add(u); it+=1
        steps.append(Step(it,u,[x for _,x in pq],sorted(visited),dist.copy(),parent.copy()))
        if u==goal: return reconstruct_path(parent,start,goal), steps, it
        for v in G.neighbors(u):
            w=G[u][v].get('weight',1.0); nd=d+w
            if nd<dist[v]:
                dist[v]=nd; parent[v]=u; heapq.heappush(pq,(nd,v))
    return [], steps, it

def astar(G,start,goal,coords):
    import heapq
    def h(a,b):
        ax,ay=coords.get(a,(0.0,0.0)); bx,by=coords.get(b,(0.0,0.0))
        return math.hypot(ax-bx, ay-by)
    g={n: math.inf for n in G.nodes}; f={n: math.inf for n in G.nodes}
    parent={start:None}; g[start]=0.0; f[start]=h(start,goal)
    open_set=[(f[start],start)]; closed=set(); steps=[]; it=0
    while open_set:
        _,u=heapq.heappop(open_set)
        if u in closed: continue
        closed.add(u); it+=1
        steps.append(Step(it,u,[x for _,x in open_set],sorted(closed),g.copy(),parent.copy()))
        if u==goal: return reconstruct_path(parent,start,goal), steps, it
        for v in G.neighbors(u):
            w=G[u][v].get('weight',1.0); t=g[u]+w
            if t<g[v]:
                g[v]=t; parent[v]=u; f[v]=t+h(v,goal); heapq.heappush(open_set,(f[v],v))
    return [], steps, it

# --- NEW: Greedy Best-First Search (heuristic-only) ---
def gbfs(G, start, goal, coords):
    import heapq
    def h(a,b):
        ax,ay=coords.get(a,(0.0,0.0)); bx,by=coords.get(b,(0.0,0.0))
        return math.hypot(ax-bx, ay-by)
    openh=[(h(start,goal), start)]
    parent={start: None}; visited=set(); steps=[]; it=0
    while openh:
        _, u = heapq.heappop(openh)
        if u in visited: 
            continue
        visited.add(u); it += 1
        steps.append(Step(it, u, [x for _,x in openh], sorted(visited),
                          {n: float('inf') for n in G.nodes}, parent.copy()))
        if u == goal:
            return reconstruct_path(parent, start, goal), steps, it
        for v in G.neighbors(u):
            if v not in visited:
                parent.setdefault(v, u)  # first time we see v
                heapq.heappush(openh, (h(v,goal), v))
    return [], steps, it

# --- NEW: Brute-Force Shortest Path (small graphs only) ---
def brute_shortest_path(G, start, goal, max_nodes=10):
    """
    Exhaustive search over all simple paths from start to goal.
    Only for small graphs (<= max_nodes) to avoid explosion.
    """
    if len(G.nodes) > max_nodes:
        # Return empty to indicate skipped due to size
        return [], [Step(0, None, [], [], {}, {})], 0
    best_path=None; best_cost=float('inf')
    steps=[]; it=0
    def dfs(u, target, visited, path, cost):
        nonlocal best_path, best_cost, it
        it += 1
        steps.append(Step(it, u, [], sorted(visited),
                          {}, {path[i]: path[i-1] if i>0 else None for i in range(len(path))}))
        if cost >= best_cost:
            return
        if u == target:
            if cost < best_cost:
                best_cost = cost
                best_path = path.copy()
            return
        for v in G.neighbors(u):
            if v not in visited:
                w = G[u][v].get('weight', 1.0)
                visited.add(v); path.append(v)
                dfs(v, target, visited, path, cost + w)
                path.pop(); visited.remove(v)
    dfs(start, goal, {start}, [start], 0.0)
    return (best_path if best_path else []), steps, it

# ===================== Other modules =====================
def apply_dynamic_changes(G, multiplier: float, affected_ratio: float=0.2):
    edges=list(G.edges()); random.shuffle(edges)
    k=max(1, int(len(edges)*affected_ratio))
    for (u,v) in edges[:k]:
        G[u][v]['weight']=max(1e-6, G[u][v].get('weight',1.0)*multiplier)

def all_pairs(G):
    length = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    return {(s,t): float(d) for s, m in length.items() for t, d in m.items()}

def greedy_assignment(vehicles, tasks, dist):
    remaining=set(vehicles); assign={}; total=0.0
    for task in tasks:
        best,bestc=None,float('inf')
        for v in remaining:
            c=dist.get((v,task), float('inf'))
            if c<bestc: bestc, best = c, v
        if best is not None:
            assign[task]=best; total+=bestc; remaining.remove(best)
    return assign, total

def dp_bnb_assignment(vehicles, tasks, dist):
    vehicles=vehicles[:len(tasks)]
    T=len(tasks); INF=float('inf')
    dp=[INF]*(1<<T); par=[(-1,-1)]*(1<<T); dp[0]=0.0
    for i in range(T):
        ndp=[INF]*(1<<T); npar=[(-1,-1)]*(1<<T)
        for s in range(1<<T):
            if dp[s]==INF: continue
            for j in range(T):
                if not (s&(1<<j)):
                    c=dist.get((vehicles[i], tasks[j]), INF)
                    ns=s|(1<<j); val=dp[s]+c
                    if val<ndp[ns]: ndp[ns]=val; npar[ns]=(s,j)
        dp,par=ndp,npar
    s=(1<<T)-1; cost=dp[s]; i=T-1; assign={}
    while s and i>=0:
        ps,j=par[s]; assign[tasks[j]]=vehicles[i]; s=ps; i-=1
    return assign, float(cost)

def tsp_nn(G, nodes):
    if not nodes: return [],0.0
    start=nodes[0]; unv=set(nodes[1:]); tour=[start]; total=0.0; cur=start
    while unv:
        best,bw=None,float('inf')
        for v in unv:
            if G.has_edge(cur,v):
                w=G[cur][v]['weight']
                if w<bw: bw=w; best=v
        if best is None: return tour, float('inf')
        total+=bw; tour.append(best); unv.remove(best); cur=best
    if G.has_edge(cur,start): total+=G[cur][start]['weight']; tour.append(start)
    else: total=float('inf')
    return tour,total

def tsp_bnb(G, nodes):
    n=len(nodes)
    if n==0: return [],0.0
    start=nodes[0]; best_tour=[]; best=float('inf')
    def bound(path, cost):
        rem=set(nodes)-set(path); b=cost
        for u in rem:
            m=min((G[u][v]['weight'] for v in nodes if u!=v and G.has_edge(u,v)), default=float('inf'))
            if m<float('inf'): b+=m
        return b
    def dfs(path,cost,unv):
        nonlocal best_tour,best
        if not unv:
            last=path[-1]
            if G.has_edge(last,start):
                tot=cost+G[last][start]['weight']
                if tot<best: best=tot; best_tour=path+[start]
            return
        if bound(path,cost)>=best: return
        u=path[-1]
        cand=sorted(list(unv), key=lambda v: G[u][v]['weight'] if G.has_edge(u,v) else float('inf'))
        for v in cand:
            if not G.has_edge(u,v): continue
            dfs(path+[v], cost+G[u][v]['weight'], unv-{v})
    dfs([start],0.0,set(nodes[1:]))
    return best_tour,best

# ===================== UI — Upload & Pick Matrix (Stateful) =====================
st.title("🚦 Smart City — Final")
st.caption("Upload Excel, select your distance matrix region (location names on both axes), then run algorithms.")

uploaded = st.file_uploader("Upload Excel (.xlsx/.xls)", type=["xlsx","xls"], key="uploader")
if uploaded and "raw_xl" not in st.session_state:
    st.session_state.raw_xl = pd.ExcelFile(uploaded)

if "matrix_ready" not in st.session_state:
    st.session_state.matrix_ready = False

if not st.session_state.matrix_ready:
    if "raw_xl" not in st.session_state:
        st.stop()

    xl = st.session_state.raw_xl
    sheet = st.selectbox("Sheet", xl.sheet_names, key="sheet")
    raw = xl.parse(sheet, header=None)
    st.write("**Sheet preview (first 25×25)**")
    st.dataframe(raw.iloc[:25, :25])

    st.markdown("### Define matrix region")
    r1 = st.number_input("Top row (0-based)", 0, max(0,len(raw)-1), 0, key="r1")
    c1 = st.number_input("Left col (0-based)", 0, max(0,raw.shape[1]-1), 0, key="c1")
    r2 = st.number_input("Bottom row (inclusive)", r1, max(0,len(raw)-1), min(len(raw)-1, r1+10), key="r2")
    c2 = st.number_input("Right col (inclusive)", c1, max(0,raw.shape[1]-1), min(raw.shape[1]-1, c1+10), key="c2")

    sub = raw.iloc[int(r1):int(r2)+1, int(c1):int(c2)+1].copy()
    st.write("**Selected region preview**")
    st.dataframe(sub)

    st.markdown("### Where are the labels?")
    use_first_row_as_columns = st.checkbox("Use FIRST ROW as column headers", True, key="use_row")
    use_first_col_as_index   = st.checkbox("Use FIRST COLUMN as row index", True, key="use_col")

    with st.form("build_form"):
        built = st.form_submit_button("✅ Build graph from this selection")
        if built:
            df = sub.copy()
            if use_first_row_as_columns:
                df.columns = [str(x).strip() for x in df.iloc[0].tolist()]
                df = df.iloc[1:].copy()
            else:
                df.columns = [str(i) for i in range(df.shape[1])]
            if use_first_col_as_index:
                df.index = [str(x).strip() for x in df.iloc[:,0].tolist()]
                df = df.iloc[:,1:].copy()
            else:
                df.index = [str(i) for i in range(df.shape[0])]

            rows = [str(x) for x in df.index]
            cols = [str(x) for x in df.columns]
            common = [r for r in rows if r in set(cols)]
            dist_df = df.loc[common, common].apply(pd.to_numeric, errors="coerce")
            np.fill_diagonal(dist_df.values, 0)

            if dist_df.shape[0] < 2 or dist_df.shape[0] != dist_df.shape[1]:
                st.error("Chosen region is not square after alignment.")
                st.stop()
            if np.isnan(dist_df.values[np.triu_indices(len(dist_df), 1)]).any():
                st.error("Some off-diagonal cells are not numeric.")
                st.stop()

            st.session_state.dist_df = dist_df
            st.session_state.directed = True
            G, coords, nodes = build_graph_from_matrix(dist_df, directed=True)
            st.session_state.G, st.session_state.coords, st.session_state.nodes = G, coords, nodes
            st.session_state.matrix_ready = True
            st.success(f"Matrix accepted: {dist_df.shape[0]} × {dist_df.shape[1]}")
            st.rerun()  # jump to main UI

# ===================== Main App (State Persists) =====================
if st.session_state.matrix_ready:
    dist_df = st.session_state.dist_df
    G = st.session_state.G
    coords = st.session_state.coords
    nodes = st.session_state.nodes

    st.success(f"Matrix accepted: {dist_df.shape[0]} × {dist_df.shape[1]}")
    with st.expander("Preview matrix"):
        st.dataframe(dist_df.iloc[:20, :20])

    directed = st.checkbox("Directed graph", value=st.session_state.get("directed", True))
    if directed != st.session_state.get("directed", True):
        st.session_state.directed = directed
        G, coords, nodes = build_graph_from_matrix(dist_df, directed=directed)
        st.session_state.G, st.session_state.coords, st.session_state.nodes = G, coords, nodes
        st.rerun()

    # ---------- Pathfinding ----------
    st.subheader("🔎 Pathfinding")
    c1,c2,c3 = st.columns(3)
    with c1: start = st.selectbox("Start", nodes, index=0, key="start")
    with c2: goal  = st.selectbox("Goal", nodes, index=min(1,len(nodes)-1), key="goal")
    with c3:
        algo  = st.selectbox(
            "Algorithm",
            ["BFS","DFS","Dijkstra","A*","Greedy Best-First","Brute-Force (small)"],
            key="algo"
        )

    if st.button("Run Algorithm", key="run_algo"):
        t0=time.perf_counter()
        if start==goal: path, steps, it = [start], [], 0
        else:
            if   algo=="BFS":                 path, steps, it = bfs(G,start,goal)
            elif algo=="DFS":                 path, steps, it = dfs(G,start,goal)
            elif algo=="Dijkstra":            path, steps, it = dijkstra(G,start,goal)
            elif algo=="A*":                  path, steps, it = astar(G,start,goal,coords)
            elif algo=="Greedy Best-First":   path, steps, it = gbfs(G,start,goal,coords)
            else:  # Brute-Force (small)
                path, steps, it = brute_shortest_path(G,start,goal,max_nodes=10)
        ms=(time.perf_counter()-t0)*1000.0
        st.session_state.last_result = {"path":path,"steps":steps,"it":it,"ms":ms}

    if "last_result" in st.session_state:
        res = st.session_state.last_result
        path, steps, it, ms = res["path"], res["steps"], res["it"], res["ms"]
        if path: st.success(f"{st.session_state.algo} found a path with {len(path)} nodes in {it} iterations ({ms:.2f} ms).")
        else:    st.error(f"{st.session_state.algo} did not find a path. ({ms:.2f} ms, {it} iterations)")
        if path: st.write("**Path:**", " → ".join(path))

        st.subheader("📈 Graph View")
        fig=plt.figure(figsize=(7,5))
        nx.draw_networkx_nodes(G, coords, node_size=500)
        nx.draw_networkx_labels(G, coords, labels={n:n for n in G.nodes}, font_size=9)
        nx.draw_networkx_edges(G, coords, arrows=True, width=1)
        if path and len(path)>1:
            pe=list(zip(path[:-1], path[1:]))
            nx.draw_networkx_nodes(G, coords, nodelist=path, node_size=600)
            nx.draw_networkx_edges(G, coords, edgelist=pe, width=3)
        st.pyplot(fig)

        st.subheader("🪜 Step-by-step trace")
        if steps and len(steps)>0 and steps[0].iteration!=0:
            idx=st.slider("Iteration",1,len(steps),1, key="trace_idx")
            s=steps[idx-1]
            L,R=st.columns(2)
            with L:
                st.markdown("**Frontier**"); st.code(", ".join(s.frontier) if s.frontier else "<empty>")
                st.markdown("**Visited**");  st.code(", ".join(s.visited) if s.visited else "<none>")
            with R:
                st.markdown("**Parent Map**"); st.json(s.parent)
                st.markdown("**Distances/Costs**"); st.json(s.distances)
        else:
            st.info("No steps to show (single-node path, brute skipped due to size, or no path).")

    # ---------- Comparison ----------
    st.subheader("📊 Algorithm Comparison (same Start/Goal)")
    def compare_algos():
        results=[]; complexities={
            'BFS':      {'time':'O(V+E)','space':'O(V)'},
            'DFS':      {'time':'O(V+E)','space':'O(V)'},
            'Dijkstra': {'time':'O((V+E) log V)','space':'O(V)'},
            'A*':       {'time':'O((V+E) log V) (h)','space':'O(V)'},
            'GBFS':     {'time':'O((V+E) log V) (h-only)','space':'O(V)'},
            'Brute':    {'time':'Exponential in V','space':'Exponential in V'},
        }
        def measure(name, fn):
            t0=time.perf_counter(); p,s,it=fn(); dt=(time.perf_counter()-t0)*1000.0
            results.append({"Algorithm":name,"Found Path?":bool(p),"Path Length (nodes)":len(p) if p else 0,
                            "Iterations":it,"Measured Time (ms)":round(dt,3),
                            "Theoretical Time":complexities[name]['time'],
                            "Theoretical Space":complexities[name]['space']})
        measure("BFS",      lambda: bfs(G,st.session_state.start,st.session_state.goal))
        measure("DFS",      lambda: dfs(G,st.session_state.start,st.session_state.goal))
        measure("Dijkstra", lambda: dijkstra(G,st.session_state.start,st.session_state.goal))
        measure("A*",       lambda: astar(G,st.session_state.start,st.session_state.goal,coords))
        measure("GBFS",     lambda: gbfs(G,st.session_state.start,st.session_state.goal,coords))
        if len(G.nodes) <= 10:
            measure("Brute",    lambda: brute_shortest_path(G,st.session_state.start,st.session_state.goal,max_nodes=10))
        return pd.DataFrame(results)

    if st.button("Run Full Comparison", key="cmp"):
        st.dataframe(compare_algos(), use_container_width=True)

    st.divider()

    # ---------- Dynamic Traffic ----------
    st.subheader("🚦 Dynamic Traffic Routing")
    dyn_ratio = st.slider("Affected edges ratio", 0.05, 1.0, 0.2, 0.05, key="dyn_ratio")
    dyn_mult  = st.slider("Weight multiplier", 0.5, 3.0, 1.5, 0.1, key="dyn_mult")
    if st.button("Apply changes & Re-run Dijkstra", key="dyn_btn"):
        apply_dynamic_changes(G, multiplier=dyn_mult, affected_ratio=dyn_ratio)
        t0=time.perf_counter()
        path_dyn, steps_dyn, it_dyn = dijkstra(G, st.session_state.start, st.session_state.goal)
        ms=(time.perf_counter()-t0)*1000.0
        if path_dyn:
            st.success(f"Dynamic Dijkstra path length {len(path_dyn)} in {it_dyn} iterations ({ms:.2f} ms).")
            st.write("Path:", " → ".join(path_dyn))
        else:
            st.error(f"No path after dynamic changes. ({ms:.2f} ms)")

    st.divider()

    # ---------- Scheduling ----------
    st.subheader("🚐 Vehicle Scheduling (Greedy vs DP/Branch-and-Bound)")
    all_nodes = list(G.nodes)
    veh = st.multiselect("Vehicle depots (locations)", all_nodes, default=all_nodes[:min(3,len(all_nodes))], key="veh")
    tsk = st.multiselect("Task nodes (locations)", all_nodes, default=all_nodes[min(3,len(all_nodes)):min(6,len(all_nodes))], key="tsk")
    if st.button("Compute Assignments", key="assign"):
        dmap = all_pairs(G)
        g_assign, g_cost = greedy_assignment(veh, tsk, dmap)
        if len(tsk) <= 12 and len(veh) >= len(tsk) and len(tsk) > 0:
            dp_assign, dp_cost = dp_bnb_assignment(veh, tsk, dmap); opt_note="(optimal exact)"
        else:
            dp_assign, dp_cost, opt_note = {}, float('nan'), "(skipped: size too large)"
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Greedy Assignment**"); st.json(g_assign)
            st.write(f"Total cost: **{round(g_cost,2)}**")
        with c2:
            st.markdown(f"**DP/Branch-and-Bound Assignment** {opt_note}"); st.json(dp_assign)
            st.write(f"Total cost: **{round(dp_cost,2) if not math.isnan(dp_cost) else '—'}**")

    st.divider()

    # ---------- TSP ----------
    st.subheader("🧩 TSP (Nearest Neighbor vs Exact BnB)")
    nodes_sel = st.multiselect("Pick locations for TSP (≤10 for exact)", all_nodes, default=all_nodes[:min(6,len(all_nodes))], key="tsp_nodes")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Run NN Heuristic", key="tsp_nn"):
            tour, cost = tsp_nn(G, nodes_sel)
            if cost < float('inf'):
                st.success(f"NN tour cost: {round(cost,2)}"); st.write("Tour:", " → ".join(tour))
            else:
                st.error("Could not complete a tour.")
    with c2:
        if st.button("Run Exact BnB", key="tsp_bnb"):
            if len(nodes_sel) > 10:
                st.warning("Exact BnB limited to ≤10 locations.")
            else:
                tour, cost = tsp_bnb(G, nodes_sel)
                if cost < float('inf'):
                    st.success(f"Exact tour cost: {round(cost,2)}"); st.write("Tour:", " → ".join(tour))
                else:
                    st.error("No Hamiltonian cycle among selected nodes.")
