use bitset_fixed::BitSet;
use proconio::input;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::collections::BTreeSet;
use std::collections::BinaryHeap;
use std::time::Instant;

#[derive(PartialEq)]
enum Square {
    Road(i32),
    Object,
}

#[derive(Clone, Copy)]
enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl ToString for Dir {
    fn to_string(&self) -> String {
        match self {
            Dir::Up => "U",
            Dir::Down => "D",
            Dir::Left => "L",
            Dir::Right => "R",
        }
        .to_string()
    }
}

fn rot90(dir: Dir) -> Dir {
    match dir {
        Dir::Up => Dir::Right,
        Dir::Down => Dir::Left,
        Dir::Left => Dir::Up,
        Dir::Right => Dir::Down,
    }
}

fn rot180(dir: Dir) -> Dir {
    rot90(rot90(dir))
}

fn rot270(dir: Dir) -> Dir {
    rot90(rot180(dir))
}

fn to_diff(dir: Dir) -> (i32, i32) {
    match dir {
        Dir::Up => (-1, 0),
        Dir::Down => (1, 0),
        Dir::Left => (0, -1),
        Dir::Right => (0, 1),
    }
}

fn to_dir(num: i32) -> Dir {
    match num.rem_euclid(4) {
        0 => Dir::Up,
        1 => Dir::Right,
        2 => Dir::Down,
        3 => Dir::Left,
        _ => panic!(),
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Axis {
    Vertical,
    Horizontal,
}

fn to_axis(dir: Dir) -> Axis {
    match dir {
        Dir::Up => Axis::Vertical,
        Dir::Down => Axis::Vertical,
        Dir::Left => Axis::Horizontal,
        Dir::Right => Axis::Horizontal,
    }
}

type Board = Vec<Vec<Square>>;
type Pos = (i32, i32);

struct Problem {
    board: Board,
    pos: Pos,
}

type Answer = Vec<Dir>;

fn neighbor_dirs(board: &Board, pos: Pos) -> Vec<Dir> {
    let mut ans = Vec::new();
    if board[pos.0 as usize][pos.1 as usize] == Square::Object {
        return ans;
    }
    let n = board.len() as i32;
    for di in 0..4 {
        let dir = to_dir(di);
        let diff = to_diff(dir);
        let row = pos.0 + diff.0;
        let col = pos.1 + diff.1;
        if row < 0 || row >= n || col < 0 || col >= n {
            continue;
        }
        if board[row as usize][col as usize] == Square::Object {
            continue;
        }
        ans.push(dir);
    }
    ans
}

fn visibles(board: &Board, pos: Pos) -> Vec<Pos> {
    let mut ans = Vec::new();
    if board[pos.0 as usize][pos.1 as usize] == Square::Object {
        return ans;
    }
    ans.push(pos);
    let n = board.len() as i32;
    for di in 0..4 {
        let dir = to_dir(di);
        let diff = to_diff(dir);
        for j in 1..n {
            let row = pos.0 + diff.0 * j;
            let col = pos.1 + diff.1 * j;
            if row < 0 || row >= n || col < 0 || col >= n {
                break;
            }
            if board[row as usize][col as usize] == Square::Object {
                break;
            }
            ans.push((row, col));
        }
    }
    ans
}

fn crossings(board: &Board) -> Vec<Pos> {
    let n = board.len() as i32;
    let mut ans = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let pos = (i, j);
            let vn = neighbor_dirs(board, pos);
            let mut sa = BTreeSet::new();
            for d in vn {
                sa.insert(to_axis(d));
            }
            if sa.len() == 2 {
                ans.push(pos);
            }
        }
    }
    ans
}

fn to_id(pos: Pos, n: usize) -> usize {
    pos.0 as usize * n + pos.1 as usize
}

fn shortest(board: &Board, pos: Pos) -> Vec<i32> {
    let mut heap = BinaryHeap::new();
    let n = board.len();
    let mut ans = vec![1_000_000_000; n * n];
    heap.push((pos, 0));
    ans[to_id(pos, n)] = 0;
    while !heap.is_empty() {
        let ((row, col), cost) = heap.pop().unwrap();
        if cost > ans[to_id((row, col), n)] {
            continue;
        }
        for i in 0..4 {
            let dir = to_dir(i);
            let (dr, dc) = to_diff(dir);
            let nr = row + dr;
            let nc = col + dc;
            if nr < 0 || nr >= n as i32 || nc < 0 || nc >= n as i32 {
                continue;
            }
            let l = match board[nr as usize][nc as usize] {
                Square::Object => continue,
                Square::Road(c) => c,
            };
            let np = (nr, nc);
            let nid = to_id(np, n);
            if ans[nid] > cost + l {
                ans[nid] = cost + l;
                heap.push((np, ans[nid]));
            }
        }
    }
    ans
}

fn needs(board: &Board) -> Vec<Pos> {
    let mut need = Vec::new();
    let n = board.len();
    for i in 0..n {
        for j in 0..n {
            if board[i][j] != Square::Object {
                need.push((i as i32, j as i32));
            }
        }
    }
    need
}

fn gen_path(board: &Board, from: Pos, mut to: Pos, cost: &[i32]) -> Vec<Dir> {
    let mut ans = Vec::new();
    let n = board.len();
    while from != to {
        for i in 0..4 {
            let dir = to_dir(i);
            let (dr, dc) = to_diff(dir);
            let nr = to.0 + dr;
            let nc = to.1 + dc;
            if nr < 0 || nr >= n as i32 || nc < 0 || nc >= n as i32 {
                continue;
            }
            if board[nr as usize][nc as usize] == Square::Object {
                continue;
            }
            let l = match board[to.0 as usize][to.1 as usize] {
                Square::Road(c) => c,
                _ => panic!(),
            };
            let np = (nr, nc);
            let nid = to_id(np, n);
            if cost[nid] + l == cost[to_id(to, n)] {
                ans.push(rot180(dir));
                to = np;
                break;
            }
        }
    }
    ans.reverse();
    ans
}

fn insert_nearest_method(mat: &[Vec<i32>]) -> Vec<usize> {
    let n = mat.len();
    let mut ans = Vec::new();
    ans.push(0);
    {
        let mut min_d = 1_000_000_000;
        let mut min_i = 0;
        for i in 1..n {
            if mat[0][i] < min_d {
                min_d = mat[0][i];
                min_i = i;
            }
        }
        ans.push(min_i);
    }
    let mut used = vec![false; n];
    used[0] = true;
    used[ans[1]] = true;
    for _rep in 0..(n - 2) {
        let mut min_d = 1_000_000_000;
        let mut min_i = 0;
        let mut min_id = 0;
        for i in 1..n {
            if used[i] {
                continue;
            }
            let mut d = 1_000_000_000;
            let mut mid = 0;
            for (id, &j) in ans.iter().enumerate() {
                if mat[i][j] < d {
                    d = mat[i][j];
                    mid = id;
                }
            }
            if d < min_d {
                min_d = d;
                min_i = i;
                min_id = mid;
            }
        }
        let m = ans.len();
        let pid = (min_id + m - 1) % m;
        let nid = (min_id + 1) % m;
        if mat[min_i][pid] < mat[min_i][nid] {
            if min_id > 0 {
                ans.insert(min_id, min_i);
            } else {
                ans.push(min_i);
            }
        } else {
            ans.insert(min_id + 1, min_i);
        }
        used[min_i] = true;
    }
    ans.push(0);
    assert!(ans.len() == n + 1);
    ans
}

fn two_opt(mat: &[Vec<i32>], ans: &mut [usize], rng: &mut SmallRng) {
    let n = mat.len();
    let dist = Uniform::from(1..n);
    let mut i = dist.sample(rng);
    let mut k = dist.sample(rng);
    if i > k {
        std::mem::swap(&mut i, &mut k);
    }
    let p1 = ans[i - 1];
    let p2 = ans[i];
    let p3 = ans[k];
    let p4 = ans[k + 1];
    if mat[p1][p3] + mat[p2][p4] >= mat[p1][p2] + mat[p3][p4] {
        return;
    }
    ans[i..=k].reverse();
}

fn optimize_tsp_impl(mat: &[Vec<i32>], rng: &mut SmallRng) -> Vec<usize> {
    let mut ans = insert_nearest_method(mat);
    for _i in 0..10000 {
        two_opt(mat, &mut ans, rng);
    }
    ans
}

fn optimize_tsp(
    board: &Board,
    all_costs: &[Vec<i32>],
    points: &[Pos],
    checkpoints: &[usize],
    rng: &mut SmallRng,
) -> Vec<usize> {
    let n = board.len();
    let mut mat = Vec::new();
    for &pid in checkpoints {
        let mut v = Vec::new();
        for &qid in checkpoints {
            v.push(all_costs[pid][to_id(points[qid], n)]);
        }
        mat.push(v);
    }
    optimize_tsp_impl(&mat, rng)
}

fn to_bitset(vpos: &[Pos], n: usize) -> BitSet {
    let mut ans = BitSet::new(n * n);
    for &pos in vpos {
        ans.set(to_id(pos, n), true);
    }
    ans
}

fn solve(prob: &Problem) -> Answer {
    let start = Instant::now();
    let crs = crossings(&prob.board);
    let mut pts = crs.clone();
    pts.push(prob.pos);
    let mut all_costs = Vec::new();
    let mut all_visibles = Vec::<BitSet>::new();
    let n = prob.board.len();
    for &p in pts.iter() {
        let c = shortest(&prob.board, p);
        all_costs.push(c);
        let vis = visibles(&prob.board, p);
        all_visibles.push(to_bitset(&vis, n));
    }
    let need: BitSet = to_bitset(&needs(&prob.board), n) & &!all_visibles.last().unwrap();
    let mut best_ans = Vec::new();
    let mut best_score = 1_000_000_000;
    let mut rng = SmallRng::from_entropy();
    let mut cnt = 0;
    loop {
        if start.elapsed().as_millis() > 2500 {
            break;
        }
        cnt += 1;
        if cnt % 100 == 0 {
            eprintln!("{} {}", cnt, best_score);
        }
        let mut current_id = crs.len();
        let mut need = need.clone();
        let mut useless = vec![false; pts.len()];
        let mut checkpoints = Vec::new();
        checkpoints.push(current_id);
        while need.count_ones() > 0 {
            let mut candidates = Vec::with_capacity(pts.len());
            for (i, v) in all_visibles.iter().enumerate() {
                if useless[i] {
                    continue;
                }
                let is = &need & v;
                let size = is.count_ones();
                if size == 0 {
                    useless[i] = true;
                    continue;
                }
                //candidates.push(i);
                //let size = need.intersection(v).len();
                candidates.push((size, i));
            }
            let i = candidates
                .choose_weighted(&mut rng, |item| item.0)
                .unwrap()
                .1;
            //let &i = candidates.choose(&mut rng).unwrap();
            useless[i] = true;
            need ^= &(&need & &all_visibles[i]);
            current_id = i;
            checkpoints.push(current_id);
        }
        let order = optimize_tsp(&prob.board, &all_costs, &pts, &checkpoints, &mut rng);
        let m = order.len();
        let mut length = 0;
        for i in 0..(m - 1) {
            let cf = order[i];
            let ct = order[i + 1];
            let from = checkpoints[cf];
            let to = checkpoints[ct];
            length += all_costs[from][to_id(pts[to], prob.board.len())];
        }
        if length >= best_score {
            continue;
        }
        let mut ans = Vec::new();
        for i in 0..(m - 1) {
            let cf = order[i];
            let ct = order[i + 1];
            let from = checkpoints[cf];
            let to = checkpoints[ct];
            ans.extend(gen_path(&prob.board, pts[from], pts[to], &all_costs[from]));
        }
        best_ans = ans;
        best_score = length;
    }
    best_ans
}

fn parse_input() -> Problem {
    input! {
        n: i32,
        si: i32,
        sj: i32,
    }
    let pos: Pos = (si, sj);
    let mut board = Board::new();
    for _i in 0..n {
        input! {
            line: String,
        }
        let mut v = Vec::new();
        for ch in line.chars() {
            if ch == '#' {
                v.push(Square::Object);
            } else {
                v.push(Square::Road(ch.to_digit(10).unwrap() as i32));
            }
        }
        board.push(v);
    }
    Problem { board, pos }
}

fn score(prob: &Problem, ans: &Answer) -> i32 {
    let mut observed = BTreeSet::new();
    let mut pos = prob.pos;
    for p in visibles(&prob.board, pos) {
        observed.insert(p);
    }
    let n = prob.board.len() as i32;
    let mut time = 0;
    for &dir in ans {
        let (dr, dc) = to_diff(dir);
        pos.0 += dr;
        pos.1 += dc;
        if pos.0 < 0 || pos.0 >= n || pos.1 < 0 || pos.1 >= n {
            eprintln!("Out of board");
            return 0;
        }
        match prob.board[pos.0 as usize][pos.1 as usize] {
            Square::Object => {
                eprintln!("Move to object");
                return 0;
            }
            Square::Road(cost) => time += cost,
        }
        for p in visibles(&prob.board, pos) {
            observed.insert(p);
        }
    }
    if pos != prob.pos {
        eprintln!("Need to return the initial point");
        return 0;
    }
    let need = needs(&prob.board);
    let score_f = if observed.len() < need.len() {
        1e4 * (observed.len() as f64) / (need.len() as f64)
    } else {
        1e4 + 1e7 * (n as f64) / (time as f64)
    };
    score_f.round() as i32
}

fn main() {
    let prob = parse_input();
    let ans = solve(&prob);
    let mut ans_str = String::with_capacity(ans.len());
    eprintln!("Score: {}", score(&prob, &ans));
    for d in ans {
        ans_str.push_str(&d.to_string());
    }
    println!("{}", ans_str);
}
