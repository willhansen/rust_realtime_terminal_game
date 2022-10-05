extern crate derive_more;
extern crate geo;
extern crate rand;

use crate::{Block, RADIUS_OF_EXACTLY_TOUCHING_ZONE};
use derive_more::{Add, Div, Mul, Sub};
use geo::algorithm::euclidean_distance::EuclideanDistance;
use geo::algorithm::line_intersection::{line_intersection, LineIntersection};
use geo::{point, CoordNum, Line, Point};
use num::traits::Pow;
use num::{clamp, ToPrimitive};
use ordered_float::OrderedFloat;
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use std::cmp::max;
use std::collections::HashMap;
use std::f32::consts::TAU;

pub type FPoint = Point<f32>;
pub type IPoint = Point<i32>;
pub type ILine = Line<i32>;
pub type FLine = Line<f32>;

pub const CLOCKWISE: f32 = 1.0;
pub const COUNTER_CLOCKWISE: f32 = -1.0;

pub fn p<T: 'static>(x: T, y: T) -> Point<T>
where
    T: CoordNum,
{
    return point!(x: x, y: y);
}

pub fn radial(r: f32, radians: f32) -> Point<f32> {
    p(r * radians.cos(), r * radians.sin())
}

pub fn right_f() -> Point<f32> {
    p(1.0, 0.0)
}
#[allow(dead_code)]
pub fn left_f() -> Point<f32> {
    p(-1.0, 0.0)
}
pub fn up_f() -> Point<f32> {
    p(0.0, 1.0)
}
#[allow(dead_code)]
pub fn down_f() -> Point<f32> {
    p(0.0, -1.0)
}
#[allow(dead_code)]
pub fn zero_f() -> Point<f32> {
    p(0.0, 0.0)
}
pub fn up_right_f() -> Point<f32> {
    up_f() + right_f()
}
pub fn up_left_f() -> Point<f32> {
    up_f() + left_f()
}
pub fn down_right_f() -> Point<f32> {
    down_f() + right_f()
}
pub fn down_left_f() -> Point<f32> {
    down_f() + left_f()
}

pub fn right_i() -> IPoint {
    p(1, 0)
}
pub fn left_i() -> IPoint {
    p(-1, 0)
}
pub fn up_i() -> IPoint {
    p(0, 1)
}
#[allow(dead_code)]
pub fn down_i() -> IPoint {
    p(0, -1)
}
#[allow(dead_code)]
pub fn zero_i() -> IPoint {
    p(0, 0)
}

pub struct SquarecastOptions {
    pub start_pos: FPoint,
    pub end_pos: FPoint,
    pub moving_square_side_length: f32,
    pub block_filter: BlockFilter,
    pub particle_location_map: Option<ParticleLocationMap>,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct SquarecastResult {
    pub start_pos: FPoint,
    pub unrounded_collider_pos: FPoint,
    pub collider_pos: FPoint,
    pub collision_normal: Option<IPoint>,
    pub collided_block_square: Option<IPoint>,
    pub collided_particle_index: Option<usize>,
}

impl SquarecastResult {
    pub fn hit_something(&self) -> bool {
        self.collided_block_square.is_some() || self.collided_particle_index.is_some()
    }
    pub fn hit_block(&self) -> bool {
        self.collided_block_square.is_some()
    }

    pub fn distance(&self) -> f32 {
        magnitude(self.collider_pos - self.start_pos)
    }

    pub fn no_hit_result(start_point: FPoint, end_point: FPoint) -> SquarecastResult {
        SquarecastResult {
            start_pos: start_point,
            unrounded_collider_pos: end_point,
            collider_pos: end_point,
            collision_normal: None,
            collided_block_square: None,
            collided_particle_index: None,
        }
    }
}

// TODO: use
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum EdgeExtension {
    Extended,
    Neutral,
    Retracted,
}

pub type LocalBlockOccupancy = [[bool; 3]; 3];
// TODO: use
pub type SquareEdgeExtensions = [[EdgeExtension; 3]; 3];

pub type ParticleLocationMap = HashMap<Point<i32>, Vec<usize>>;

pub type BlockFilter = Box<dyn Fn(Block) -> bool>;

#[allow(dead_code)]
pub fn single_block_squarecast(
    start_point: Point<f32>,
    end_point: Point<f32>,
    grid_square_center: Point<i32>,
    moving_square_side_length: f32,
) -> SquarecastResult {
    single_block_squarecast_with_edge_extensions(
        start_point,
        end_point,
        grid_square_center,
        moving_square_side_length,
        [[false; 3]; 3],
    )
}

pub fn first_hit(collisions: Vec<SquarecastResult>) -> SquarecastResult {
    let start_pos = collisions[0].start_pos;
    *collisions
        .iter()
        .min_by_key(|r| OrderedFloat(r.collider_pos.euclidean_distance(&start_pos)))
        .unwrap()
}

pub fn visible_xy_to_actual_xy<T: Copy>(a: [[T; 3]; 3]) -> [[T; 3]; 3] {
    // visible
    // [ [ 7, 8, 9 ],
    //   [ 4, 5, 6 ],
    //   [ 1, 2, 3 ] ]
    //
    // but for [x][y] indexes, we need
    //
    // [ [ 1, 4, 7 ],
    //   [ 2, 5, 8 ],
    //   [ 3, 6, 9 ] ]
    [
        [a[2][0], a[1][0], a[0][0]],
        [a[2][1], a[1][1], a[0][1]],
        [a[2][2], a[1][2], a[0][2]],
    ]
}

pub fn single_block_linecast_with_filled_cracks(
    start_point: Point<f32>,
    end_point: Point<f32>,
    grid_square_center: Point<i32>,
    adjacent_squares_occupied: LocalBlockOccupancy,
) -> SquarecastResult {
    single_block_squarecast_with_edge_extensions(
        start_point,
        end_point,
        grid_square_center,
        0.0,
        adjacent_squares_occupied,
    )
}

pub fn single_block_squarecast_with_edge_extensions(
    start_point: Point<f32>,
    end_point: Point<f32>,
    grid_square_center: Point<i32>,
    moving_square_side_length: f32,
    adjacent_squares_occupied: LocalBlockOccupancy,
) -> SquarecastResult {
    let default_result = SquarecastResult::no_hit_result(start_point, end_point);
    // formulates the problem as a point crossing the boundary of an r=1 square
    let movement_line = geo::Line::new(start_point, end_point);
    //println!("movement_line: {:?}", movement_line);
    let combined_square_side_length = 1.0 + moving_square_side_length;

    let counter_clockwise_collision_edges =
        get_counter_clockwise_collision_edges_with_known_adjacency(
            floatify(grid_square_center),
            combined_square_side_length,
            adjacent_squares_occupied,
        );

    // You can define structs IN a function!?  So convenient!
    #[derive(Copy, Clone, PartialEq, Debug)]
    struct EdgeCollision {
        edge: geo::Line<f32>,
        point_of_impact: FPoint,
        normal: IPoint,
    }

    //println!("expanded_edges: {:?}", expanded_square_edges);
    let mut candidate_edge_collisions = Vec::<EdgeCollision>::new();
    for edge in counter_clockwise_collision_edges {
        if let Some(LineIntersection::SinglePoint {
            intersection: coord,
            is_proper: _,
        }) = line_intersection(movement_line, edge)
        {
            let edge_as_vect: FPoint = (edge.end - edge.start).into();
            let normal = snap_to_grid(direction(quarter_turns_counter_clockwise(edge_as_vect, 3)));
            candidate_edge_collisions.push(EdgeCollision {
                edge,
                point_of_impact: coord.into(),
                normal,
            });
        }
    }
    if candidate_edge_collisions.is_empty() {
        return default_result;
    }

    // four intersections with extended walls of stationary square
    candidate_edge_collisions.sort_by(|a, b| {
        start_point
            .euclidean_distance(&a.point_of_impact)
            .partial_cmp(&start_point.euclidean_distance(&b.point_of_impact))
            .unwrap()
    });
    //println!("intersections: {:?}", candidate_edge_intersections);
    let true_collision_point = candidate_edge_collisions[0].point_of_impact;
    let collision_normal = candidate_edge_collisions[0].normal;
    let rounded_collision_point = snap_to_square_perimeter_with_forbidden_sides(
        true_collision_point,
        floatify(grid_square_center),
        combined_square_side_length,
        adjacent_squares_occupied,
    );

    SquarecastResult {
        start_pos: start_point,
        unrounded_collider_pos: true_collision_point,
        collider_pos: rounded_collision_point,
        collision_normal: Some(collision_normal),
        collided_block_square: Some(grid_square_center),
        collided_particle_index: None,
    }
}

pub fn get_counter_clockwise_collision_edges_with_known_adjacency(
    center: FPoint,
    nominal_side_length: f32,
    adjacency_map: LocalBlockOccupancy,
) -> Vec<geo::Line<f32>> {
    // ┌────────────────┐
    // │                │
    // ├────────────────┼────┐
    // │                │    │
    // │                │    │
    // │                │    │
    // │                │    │
    // │                │    │
    // │                │    │
    // ├────────────────┼────┘
    // │                │
    // └────────────────┘

    let inner_halflength = nominal_side_length / 2.0 - RADIUS_OF_EXACTLY_TOUCHING_ZONE;
    let outer_halflength = nominal_side_length / 2.0 + RADIUS_OF_EXACTLY_TOUCHING_ZONE;

    let mut output_lines = Vec::<FLine>::new();
    // for every orthogonal direction
    for i in 0..4 {
        let this_edge_dir = orthogonal_direction(i);
        let left_corner_dir = this_edge_dir + quarter_turns_counter_clockwise(this_edge_dir, 1);
        let right_corner_dir = this_edge_dir + quarter_turns_counter_clockwise(this_edge_dir, 3);
        let left_edge_dir = quarter_turns_counter_clockwise(this_edge_dir, 1);
        let right_edge_dir = quarter_turns_counter_clockwise(this_edge_dir, 3);

        let dir_to_expansion_val =
            |v: IPoint| -> bool { adjacency_map[(v.x() + 1) as usize][(v.y() + 1) as usize] };

        let block_at_this_edge = dir_to_expansion_val(this_edge_dir);
        let block_at_left_edge = dir_to_expansion_val(left_edge_dir);
        let block_at_right_edge = dir_to_expansion_val(right_edge_dir);
        let block_at_left_corner = dir_to_expansion_val(left_corner_dir);
        let block_at_right_corner = dir_to_expansion_val(right_corner_dir);

        let should_expand_this_edge = block_at_this_edge;
        let should_expand_left_edge = block_at_left_edge;
        let should_expand_right_edge = block_at_right_edge;

        let should_expand_right_corner =
            should_expand_this_edge && should_expand_right_edge && block_at_right_corner;
        let should_expand_left_corner =
            should_expand_this_edge && should_expand_left_edge && block_at_left_corner;

        let need_outer_edge =
            should_expand_this_edge || should_expand_right_corner || should_expand_left_corner;

        let need_inner_edge = !should_expand_this_edge
            || (!should_expand_right_corner && should_expand_right_edge)
            || (!should_expand_left_corner && should_expand_left_edge);

        let wide_right_corner = center
            + floatify(this_edge_dir) * inner_halflength
            + floatify(right_edge_dir) * outer_halflength;
        let wide_left_corner = center
            + floatify(this_edge_dir) * inner_halflength
            + floatify(left_edge_dir) * outer_halflength;
        let tall_right_corner = center
            + floatify(this_edge_dir) * outer_halflength
            + floatify(right_edge_dir) * inner_halflength;
        let tall_left_corner = center
            + floatify(this_edge_dir) * outer_halflength
            + floatify(left_edge_dir) * inner_halflength;
        let far_right_corner = center + floatify(right_corner_dir) * outer_halflength;
        let far_left_corner = center + floatify(left_corner_dir) * outer_halflength;
        let near_right_corner = center + floatify(right_corner_dir) * inner_halflength;
        let near_left_corner = center + floatify(left_corner_dir) * inner_halflength;

        if need_outer_edge {
            let start_point = if should_expand_right_corner {
                far_right_corner
            } else {
                tall_right_corner
            };
            let end_point = if should_expand_left_corner {
                far_left_corner
            } else {
                tall_left_corner
            };
            let far_collision_edge = geo::Line::new(start_point, end_point);
            output_lines.push(far_collision_edge);
        }

        if need_inner_edge {
            // Need to start with right side so output lines are counter-clockwise
            let start_point = if should_expand_right_edge {
                wide_right_corner
            } else {
                near_right_corner
            };
            let end_point = if should_expand_left_edge {
                wide_left_corner
            } else {
                near_left_corner
            };
            let close_collision_edge = geo::Line::new(start_point, end_point);
            output_lines.push(close_collision_edge);
        }
    }
    output_lines
}

pub fn single_block_linecast(
    start_point: Point<f32>,
    end_point: Point<f32>,
    grid_square_center: Point<i32>,
) -> SquarecastResult {
    single_block_squarecast(start_point, end_point, grid_square_center, 0.0)
}

pub fn single_block_unit_squarecast(
    start_point: Point<f32>,
    end_point: Point<f32>,
    grid_square_center: Point<i32>,
) -> SquarecastResult {
    single_block_squarecast(start_point, end_point, grid_square_center, 1.0)
}

pub fn orthogonal_direction<T: 'static + CoordNum + num::Signed + std::fmt::Display>(
    dir_num: T,
) -> Point<T> {
    let dir_num_int = dir_num.to_i32().unwrap() % 4;
    quarter_turns_counter_clockwise(Point::<T>::new(T::one(), T::zero()), dir_num_int)
}
pub fn diagonal_direction<T: 'static + CoordNum + num::Signed + std::fmt::Display>(
    dir_num: T,
) -> Point<T> {
    let dir_num_int = dir_num.to_i32().unwrap() % 4;
    quarter_turns_counter_clockwise(Point::<T>::new(T::one(), T::one()), dir_num_int)
}

pub fn adjacent_direction<T: CoordNum + num::Signed + std::fmt::Display>(dir_num: T) -> Point<T> {
    let dir_num_int = dir_num.to_i32().unwrap() % 8;
    match dir_num_int {
        0 => Point::<T>::new(T::one(), T::zero()),
        1 => Point::<T>::new(T::one(), T::one()),
        2 => Point::<T>::new(T::zero(), T::one()),
        3 => Point::<T>::new(-T::one(), T::one()),
        4 => Point::<T>::new(-T::one(), T::zero()),
        5 => Point::<T>::new(-T::one(), -T::one()),
        6 => Point::<T>::new(T::zero(), -T::one()),
        7 => Point::<T>::new(T::one(), -T::one()),
        _ => panic!("bad direction number: {}", dir_num),
    }
}

pub fn round_to_direction_number(point: Point<f32>) -> i32 {
    let (x, y) = point.x_y();
    if x.abs() > y.abs() {
        if x > 0.0 {
            return 0;
        } else {
            return 2;
        }
    } else {
        if y > 0.0 {
            return 1;
        } else {
            return 3;
        }
    }
}
pub fn grid_squares_overlapped_by_floating_unit_square(pos: Point<f32>) -> Vec<Point<i32>> {
    grid_squares_overlapped_by_floating_square(pos, 1.0)
}

pub fn grid_squares_touched_by_floating_square_with_tolerance(
    pos: Point<f32>,
    square_side_length: f32,
    this_much_overlap_required: f32,
) -> Vec<Point<i32>> {
    let bottom_left_point = pos - p(1.0, 1.0) * square_side_length / 2.0;
    let top_right_point = pos + p(1.0, 1.0) * square_side_length / 2.0;

    let nudge = p(this_much_overlap_required, this_much_overlap_required);

    let bottom_left_square = round_vector_with_tie_break_toward_inf(bottom_left_point + nudge);
    let top_right_square = round_vector_with_tie_break_toward_neg_inf(top_right_point - nudge);
    let mut output = Vec::<Point<i32>>::new();
    for x in bottom_left_square.x()..=top_right_square.x() {
        for y in bottom_left_square.y()..=top_right_square.y() {
            output.push(p(x, y));
        }
    }
    output
}

pub fn grid_squares_touched_or_overlapped_by_floating_square(
    pos: Point<f32>,
    square_side_length: f32,
) -> Vec<Point<i32>> {
    grid_squares_touched_by_floating_square_with_tolerance(
        pos,
        square_side_length,
        -RADIUS_OF_EXACTLY_TOUCHING_ZONE,
    )
}

pub fn grid_squares_overlapped_by_floating_square(
    pos: Point<f32>,
    square_side_length: f32,
) -> Vec<Point<i32>> {
    grid_squares_touched_by_floating_square_with_tolerance(
        pos,
        square_side_length,
        RADIUS_OF_EXACTLY_TOUCHING_ZONE,
    )
}

pub fn snap_to_grid(world_pos: Point<f32>) -> Point<i32> {
    return round(world_pos);
}
pub fn offset_from_grid(world_pos: Point<f32>) -> Point<f32> {
    return world_pos - floatify(snap_to_grid(world_pos));
}

pub fn point_inside_grid_square(point: Point<f32>, square: Point<i32>) -> bool {
    let diff = point - floatify(square);
    let within_x_limit = diff.x().abs() < 0.5 - RADIUS_OF_EXACTLY_TOUCHING_ZONE;
    let within_y_limit = diff.y().abs() < 0.5 - RADIUS_OF_EXACTLY_TOUCHING_ZONE;
    within_x_limit && within_y_limit
}

pub fn point_exactly_touching_unit_square(point: Point<f32>, square: Point<i32>) -> bool {
    point_exactly_touching_square(point, square, 1.0)
}

pub fn point_exactly_touching_square(
    point: Point<f32>,
    square: Point<i32>,
    square_side_length: f32,
) -> bool {
    let diff = point - floatify(square);
    let on_x_limit =
        (diff.x().abs() - square_side_length / 2.0).abs() <= RADIUS_OF_EXACTLY_TOUCHING_ZONE;
    let on_y_limit =
        (diff.y().abs() - square_side_length / 2.0).abs() <= RADIUS_OF_EXACTLY_TOUCHING_ZONE;
    let within_x_limit =
        diff.x().abs() < square_side_length / 2.0 - RADIUS_OF_EXACTLY_TOUCHING_ZONE;
    let within_y_limit =
        diff.y().abs() < square_side_length / 2.0 - RADIUS_OF_EXACTLY_TOUCHING_ZONE;
    return (within_x_limit && on_y_limit)
        || (within_y_limit && on_x_limit)
        || (on_x_limit && on_y_limit);
}

pub fn point_diagonally_touching_square(
    point: Point<f32>,
    square: Point<i32>,
    square_side_length: f32,
) -> bool {
    let diff = point - floatify(square);
    let on_x_limit =
        (diff.x().abs() - square_side_length / 2.0).abs() <= RADIUS_OF_EXACTLY_TOUCHING_ZONE;
    let on_y_limit =
        (diff.y().abs() - square_side_length / 2.0).abs() <= RADIUS_OF_EXACTLY_TOUCHING_ZONE;
    on_x_limit && on_y_limit
}

#[allow(dead_code)]
pub fn trunc(vec: Point<f32>) -> Point<i32> {
    return Point::<i32>::new(vec.x().trunc() as i32, vec.y().trunc() as i32);
}

pub fn round(vec: Point<f32>) -> Point<i32> {
    return Point::<i32>::new(vec.x().round() as i32, vec.y().round() as i32);
}

pub fn round_vector_with_tie_break_toward_inf(vec: Point<f32>) -> Point<i32> {
    return Point::<i32>::new(
        round_with_tie_break_toward_inf(vec.x()),
        round_with_tie_break_toward_inf(vec.y()),
    );
}

pub fn round_vector_with_tie_break_toward_neg_inf(vec: Point<f32>) -> Point<i32> {
    return Point::<i32>::new(
        round_with_tie_break_toward_neg_inf(vec.x()),
        round_with_tie_break_toward_neg_inf(vec.y()),
    );
}
pub fn round_with_tie_break_toward_inf(x: f32) -> i32 {
    if (x - x.round()).abs() == 0.5 {
        return (x + 0.1).round() as i32;
    } else {
        return x.round() as i32;
    }
}
pub fn round_with_tie_break_toward_neg_inf(x: f32) -> i32 {
    if (x - x.round()).abs() == 0.5 {
        return (x - 0.1).round() as i32;
    } else {
        return x.round() as i32;
    }
}

pub fn floatify(vec: Point<i32>) -> Point<f32> {
    return Point::<f32>::new(vec.x() as f32, vec.y() as f32);
}

pub fn magnitude(vec: Point<f32>) -> f32 {
    return (vec.x().pow(2.0) + vec.y().pow(2.0)).sqrt();
}
pub fn magnitude_squared(vec: Point<f32>) -> f32 {
    return vec.x().pow(2.0) + vec.y().pow(2.0);
}
pub fn direction(vec: Point<f32>) -> Point<f32> {
    if vec == zero_f() || magnitude(vec) == 0.0 {
        return zero_f();
    }
    return vec / magnitude(vec);
}
// because convenient
pub fn normalized(vec: Point<f32>) -> Point<f32> {
    direction(vec)
}

#[allow(dead_code)]
pub fn fract(vec: Point<f32>) -> Point<f32> {
    return Point::<f32>::new(vec.x().fract(), vec.y().fract());
}

pub fn sign<T>(p: Point<T>) -> Point<T>
where
    T: SignedExt + CoordNum,
{
    return Point::<T>::new(p.x().sign(), p.y().sign());
}

pub trait SignedExt: num::Signed {
    fn sign(&self) -> Self;
}

impl<T: num::Signed> SignedExt for T {
    // I am so angry this is not built-in
    fn sign(&self) -> T {
        return if *self == T::zero() {
            T::zero()
        } else if self.is_negative() {
            -T::one()
        } else {
            T::one()
        };
    }
}

pub fn project_a_onto_b(v1: Point<f32>, v2: Point<f32>) -> Point<f32> {
    return direction(v2) * v1.dot(v2) / magnitude(v2);
}

pub trait PointExt {
    fn add_assign(&mut self, rhs: Self);
}

impl<T: CoordNum> PointExt for Point<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

pub fn decelerate_linearly_to_cap(
    start_vel: f32,
    cap_vel: f32,
    deceleration_above_cap: f32,
) -> f32 {
    if start_vel.abs() < cap_vel {
        return start_vel;
    }
    let accel_direction = -start_vel.sign();
    let dv = accel_direction * deceleration_above_cap;
    let crossed_zero = start_vel.sign() * (start_vel + dv).sign() < 0.0;
    if (start_vel + dv).abs() < cap_vel || crossed_zero {
        return start_vel.sign() * cap_vel;
    }
    return start_vel + dv;
}

pub fn accelerate_within_max_speed(
    start_vel: f32,
    desired_direction: i32,
    max_speed: f32,
    acceleration: f32,
) -> f32 {
    let want_speed_up = desired_direction.sign() as f32 * start_vel.sign() == 1.0;
    let start_too_fast = start_vel.abs() > max_speed;

    if start_too_fast && want_speed_up {
        start_vel
    } else {
        accelerate_to_target_vel(
            start_vel,
            (desired_direction as f32) * max_speed,
            acceleration,
        )
    }
}

pub fn accelerate_to_target_vel(start_vel: f32, target_vel: f32, acceleration: f32) -> f32 {
    let dir = (target_vel - start_vel).sign();
    let dv = acceleration * dir;
    let possible_new_vel = start_vel + dv;
    if dir > 0.0 {
        clamp(possible_new_vel, f32::NEG_INFINITY, target_vel)
    } else {
        clamp(possible_new_vel, target_vel, f32::INFINITY)
    }
}

pub fn world_space_to_grid_space(before: Point<f32>, vertical_stretch_factor: f32) -> Point<f32> {
    p(before.x(), before.y() / vertical_stretch_factor)
}
pub fn grid_space_to_world_space(before: Point<f32>, vertical_stretch_factor: f32) -> Point<f32> {
    p(before.x(), before.y() * vertical_stretch_factor)
}

pub fn random_direction() -> Point<f32> {
    let mut rng = rand::thread_rng();
    let dir = p(rng.gen::<f32>() - 0.5, rng.gen::<f32>() - 0.5) * 2.0;
    if dir == p(0.0, 0.0) {
        return p(1.0, 0.0);
    } else {
        return dir;
    }
}

pub fn rand_in_square(square: IPoint) -> FPoint {
    let c = floatify(square);
    p(
        rand_in_range(c.x() - 0.5, c.x() + 0.5),
        rand_in_range(c.y() - 0.5, c.y() + 0.5),
    )
}

pub fn rand_in_range<T: SampleUniform + std::cmp::PartialOrd>(start: T, end: T) -> T {
    let mut rng = rand::thread_rng();
    rng.gen_range(start..end)
}

pub fn lerp<T: std::ops::Mul<f32, Output = T> + std::ops::Add<T, Output = T>>(
    a: T,
    b: T,
    t: f32,
) -> T {
    a * (1.0 - t) + b * t
}
pub fn inverse_lerp(a: f32, b: f32, value_between_a_and_b: f32) -> f32 {
    (value_between_a_and_b - a) / (b - a)
}

pub fn floating_square_orthogonally_touching_fixed_square(
    float_square_pos: Point<f32>,
    fixed_square_pos: Point<i32>,
) -> bool {
    point_exactly_touching_square(float_square_pos, fixed_square_pos, 2.0)
        && !point_diagonally_touching_square(float_square_pos, fixed_square_pos, 2.0)
}

pub fn points_in_line_with_max_gap(
    start: Point<f32>,
    end: Point<f32>,
    max_linear_density: f32,
) -> Vec<Point<f32>> {
    let blocks = magnitude(end - start);
    let num_inner_points: i32 = (blocks * max_linear_density).ceil() as i32 - 1;
    let mut output = vec![];
    output.push(start);
    for i in 1..num_inner_points {
        output.push(lerp_2d(start, end, i as f32 / num_inner_points as f32));
    }
    output.push(end);
    return output;
}

pub fn lin_space_from_start_2d(start: FPoint, end: FPoint, density: f32) -> Vec<FPoint> {
    // end point is not included.  Start point is included

    let num_points_to_check = (magnitude(end - start) * density).floor() as i32;
    let mut points = Vec::<Point<f32>>::new();
    let dir = direction(end - start);
    for i in 0..num_points_to_check {
        points.push(start + dir * (i as f32 / density));
    }
    points
}

pub fn lerp_2d(a: Point<f32>, b: Point<f32>, t: f32) -> Point<f32> {
    p(lerp(a.x(), b.x(), t), lerp(a.y(), b.y(), t))
}
pub fn inverse_lerp_2d(a: FPoint, b: FPoint, third_point: FPoint) -> f32 {
    magnitude(third_point - a) / magnitude(b - a)
}

pub fn rotated(vect: Point<f32>, radians: f32) -> Point<f32> {
    let (x, y) = vect.x_y();
    p(
        x * radians.cos() - y * radians.sin(),
        x * radians.sin() + y * radians.cos(),
    )
}

pub fn rotated_degrees(vect: Point<f32>, degrees: f32) -> Point<f32> {
    rotated(vect, degrees.to_radians())
}

pub fn int_to_T<T: num::Signed>(x: i32) -> T {
    match x {
        1 => T::one(),
        0 => T::zero(),
        -1 => -T::one(),
        _ => panic!(),
    }
}
pub fn quarter_turns_counter_clockwise<T: 'static + CoordNum + num::Signed + std::fmt::Display>(
    vect: Point<T>,
    quarter_periods: i32,
) -> Point<T> {
    let (x, y) = vect.x_y();
    p(
        x * int_to_T(int_cos(quarter_periods)) - y * int_to_T(int_sin(quarter_periods)),
        x * int_to_T(int_sin(quarter_periods)) + y * int_to_T(int_cos(quarter_periods)),
    )
}
pub fn int_cos(quarter_periods: i32) -> i32 {
    match quarter_periods.rem_euclid(4) {
        0 => 1,
        1 | 3 => 0,
        2 => -1,
        _ => panic!(),
    }
}
pub fn int_sin(quarter_periods: i32) -> i32 {
    match quarter_periods.rem_euclid(4) {
        0 | 2 => 0,
        1 => 1,
        3 => -1,
        _ => panic!(),
    }
}
pub fn time_synchronized_points_on_line(
    start_point: Point<f32>,
    end_point: Point<f32>,
    start_time: f32,
    end_time: f32,
    time_period: f32,
) -> Vec<Point<f32>> {
    let mut output_points = vec![];
    for time_as_fraction in
        time_synchronized_interpolation_fractions(start_time, end_time, time_period)
    {
        output_points.push(lerp_2d(start_point, end_point, time_as_fraction));
    }
    output_points
}

pub fn time_synchronized_interpolation_fractions(
    start_time: f32,
    end_time: f32,
    time_period: f32,
) -> Vec<f32> {
    let start_time_in_periods = start_time / time_period;
    let end_time_in_periods = end_time / time_period;
    let first_period_in_range = start_time_in_periods.ceil() as i32;
    // intentionally leave off last period if exactly at end time
    let one_past_last_period_in_range = end_time_in_periods.ceil() as i32;
    let mut output_fractions = vec![];
    for i in first_period_in_range..one_past_last_period_in_range {
        let time = i as f32 * time_period;
        output_fractions.push(inverse_lerp(start_time, end_time, time));
    }
    output_fractions
}

// In radians
pub fn angle_between(v1: FPoint, v2: FPoint) -> f32 {
    (v1.dot(v2) / (magnitude(v1) * magnitude(v2))).acos()
}
pub fn modulo(a: f32, b: f32) -> f32 {
    ((a % b) + b) % b
}

// [0, TAU)
pub fn to_standard_angle(a: f32) -> f32 {
    modulo(a, TAU)
}

pub fn angle_of_vector(v: FPoint) -> f32 {
    v.y().atan2(v.x())
}

pub fn vector_from_radial(angle: f32, length: f32) -> FPoint {
    p(angle.cos(), angle.sin()) * length
}

pub fn shortest_delta_to_angle(start_angle: f32, end_angle: f32) -> f32 {
    let a = to_standard_angle(start_angle);
    let b = to_standard_angle(end_angle);
    let b_plus = b + TAU;
    let b_minus = b - TAU;
    let b_closest = *vec![b, b_plus, b_minus]
        .iter()
        .min_by_key(|&&b| OrderedFloat((b - a).abs()))
        .unwrap();
    b_closest - a
}
// In radians
pub fn rotate_angle_towards(start_angle: f32, target_angle: f32, max_step: f32) -> f32 {
    let diff = shortest_delta_to_angle(start_angle, target_angle);
    if max_step.abs() > diff.abs() {
        target_angle
    } else {
        start_angle + diff.sign() * max_step
    }
}

pub fn rotate_vector_towards(start: FPoint, target: FPoint, max_angle_step: f32) -> FPoint {
    let start_angle = angle_of_vector(start);
    let start_length = magnitude(start);
    let target_angle = angle_of_vector(target);
    let end_angle = rotate_angle_towards(start_angle, target_angle, max_angle_step);
    vector_from_radial(end_angle, start_length)
}

pub fn flip_x(v: FPoint) -> FPoint {
    p(-v.x(), v.y())
}
pub fn flip_y(v: FPoint) -> FPoint {
    p(v.x(), -v.y())
}

pub fn ticks_to_stop_from_speed(v0: f32, a: f32) -> Option<f32> {
    if a == 0.0 {
        None
    } else {
        Some(v0 / a)
    }
}

#[derive(Copy, Clone, PartialEq, Debug, Add, Sub, Mul, Div)]
pub struct KinematicState {
    pub pos: FPoint,
    pub vel: FPoint,
    pub accel: FPoint,
}
impl KinematicState {
    pub fn extrapolated(&self, dt: f32) -> KinematicState {
        KinematicState {
            pos: self.pos + self.vel * dt + self.accel * 0.5 * dt * dt,
            vel: self.vel + self.accel * dt,
            accel: self.accel,
        }
    }
    pub fn dt_to_slowest_point(&self) -> f32 {
        time_of_line_closest_approach_to_origin(self.accel, self.vel)
    }
    pub fn extrapolated_with_full_stop_at_slowest(&self, dt: f32) -> KinematicState {
        self.extrapolated_with_full_stop_or_linear_motion_at_slowest(dt, true)
    }
    pub fn extrapolated_with_linear_motion_at_slowest(&self, dt: f32) -> KinematicState {
        self.extrapolated_with_full_stop_or_linear_motion_at_slowest(dt, false)
    }
    pub fn extrapolated_with_full_stop_or_linear_motion_at_slowest(
        &self,
        dt: f32,
        full_stop: bool,
    ) -> KinematicState {
        let dt_to_slowest = self.dt_to_slowest_point();
        let slowest_point_is_in_past = dt_to_slowest < 0.0;
        let slowest_point_is_beyond_dt = dt < dt_to_slowest;
        let is_constant_speed = self.accel == zero_f();
        let slowest_point_is_singular = !is_constant_speed;
        let slowest_point_is_now = dt_to_slowest == 0.0;

        if slowest_point_is_in_past || slowest_point_is_beyond_dt || is_constant_speed
        //|| (slowest_point_is_singular && slowest_point_is_now)
        {
            return self.extrapolated(dt);
        }
        let mut mid_state = self.extrapolated(dt_to_slowest);
        mid_state.accel = zero_f();
        if full_stop {
            mid_state.vel = zero_f();
            return mid_state;
        }
        let remaining_dt = dt - dt_to_slowest;
        return mid_state.extrapolated(remaining_dt);
    }
    pub fn extrapolated_with_speed_cap(&self, dt: f32, speed_cap: f32) -> KinematicState {
        if magnitude(self.vel) <= speed_cap {
            if let Some(sooner_dt) =
                when_parabolic_motion_reaches_speed(self.vel, self.accel, speed_cap)
            {
                if sooner_dt < dt {
                    let mid_state = self.extrapolated(sooner_dt);
                    let remaining_dt = dt - sooner_dt;
                    return KinematicState {
                        pos: mid_state.pos + mid_state.vel * remaining_dt,
                        vel: mid_state.vel,
                        accel: mid_state.accel,
                    };
                }
            }
        }
        self.extrapolated(dt)
    }
    pub fn extrapolated_delta(&self, dt: f32) -> KinematicState {
        self.extrapolated(dt) - *self
    }
    pub fn extrapolated_delta_with_speed_cap(&self, dt: f32, speed_cap: f32) -> KinematicState {
        self.extrapolated_with_speed_cap(dt, speed_cap) - *self
    }
    pub fn extrapolated_delta_with_full_stop_at_slowest(&self, dt: f32) -> KinematicState {
        self.extrapolated_with_full_stop_at_slowest(dt) - *self
    }
    pub fn extrapolated_delta_with_linear_motion_at_slowest(&self, dt: f32) -> KinematicState {
        self.extrapolated_with_linear_motion_at_slowest(dt) - *self
    }
}

pub fn time_of_line_closest_approach_to_origin(v: FPoint, x0: FPoint) -> f32 {
    if v == zero_f() {
        return 0.0;
    }
    -v.dot(x0) / (v.dot(v))
}

pub fn when_linear_motion_hits_circle_from_inside(
    start_pos: FPoint,
    vel: FPoint,
    radius: f32,
) -> Option<f32> {
    let starts_on_circle = magnitude(start_pos) == radius;
    let moving_away_from_center_at_start = start_pos.dot(vel) > 0.0;
    if starts_on_circle && moving_away_from_center_at_start {
        return Some(0.0);
    }
    let not_moving = vel == zero_f();
    if not_moving {
        return None;
    }

    let a = magnitude_squared(vel);
    let b = 2.0 * start_pos.dot(vel);
    let c = magnitude_squared(start_pos) - radius.pow(2.0);
    let determinant = b * b - 4.0 * a * c;
    // never collide with circle
    if determinant < 0.0 {
        return None;
    }
    let t1 = (-b + determinant.sqrt()) / (2.0 * a);
    let t2 = (-b - determinant.sqrt()) / (2.0 * a);
    //dbg!(start_pos, vel, radius, a, b, c, determinant, t1, t2);
    // collided with circle in past
    if t1 < 0.0 && t2 < 0.0 {
        return None;
    }
    // two collisions in the future (now included)
    if t1 >= 0.0 && t2 >= 0.0 {
        if starts_on_circle {
            Some(t1.max(t2))
        } else {
            Some(t1.min(t2))
        }
    }
    // one collision in past, one in future (started inside circle)
    else {
        Some(t1.max(t2))
    }
}

pub fn when_parabolic_motion_reaches_speed(
    start_vel: FPoint,
    acceleration: FPoint,
    target_speed: f32,
) -> Option<f32> {
    when_linear_motion_hits_circle_from_inside(start_vel, acceleration, target_speed)
}

pub fn remove_by_value<T: PartialEq>(vec: &mut Vec<T>, val: &T) {
    vec.retain(|x| *x != *val);
}

pub fn get_3x3_squares() -> Vec<IPoint> {
    let mut output_squares = Vec::<IPoint>::new();
    for dx in -1..=1 {
        for dy in -1..=1 {
            let delta = p(dx, dy);
            output_squares.push(delta);
        }
    }
    output_squares
}

pub fn empty_local_block_occupancy() -> LocalBlockOccupancy {
    [[false; 3]; 3]
}

pub fn flip_block_occupancy_x(occupancy: LocalBlockOccupancy) -> LocalBlockOccupancy {
    let mut flipped = empty_local_block_occupancy();
    for y in 0..3 {
        for x in 0..3 {
            flipped[x][y] = occupancy[2 - x][y];
        }
    }
    flipped
}
pub fn flip_block_occupancy_y(occupancy: LocalBlockOccupancy) -> LocalBlockOccupancy {
    let mut flipped = empty_local_block_occupancy();
    for x in 0..3 {
        for y in 0..3 {
            flipped[x][y] = occupancy[x][2 - y];
        }
    }
    flipped
}

pub fn is_deep_concave_corner(candidate: LocalBlockOccupancy) -> bool {
    let eg = visible_xy_to_actual_xy([
        [true, false, false],
        [true, false, false],
        [true, true, true],
    ]);
    let possible_corners = vec![
        eg,
        flip_block_occupancy_x(eg),
        flip_block_occupancy_y(eg),
        flip_block_occupancy_x(flip_block_occupancy_y(eg)),
    ];
    possible_corners.contains(&candidate)
}
pub fn inside_direction_of_corner(corner: LocalBlockOccupancy) -> FPoint {
    let mut output_vector = zero_f();
    for i in 0..4 {
        let dir = orthogonal_direction(i as i32);
        let (x, y) = dir.x_y();
        let square_empty = corner[(x + 1) as usize][(y + 1) as usize] == false;
        if square_empty {
            output_vector.add_assign(floatify(dir));
        }
    }
    output_vector
}

// for tests
pub fn points_nearly_equal(a: Point<f32>, b: Point<f32>) -> bool {
    let result = magnitude(a - b) < 0.0001;
    if !result {
        dbg!(a, b);
    }
    result
}
pub fn nearly_equal(a: f32, b: f32) -> bool {
    let result = (a - b).abs() < 0.0001;
    if !result {
        dbg!(a, b);
    }
    result
}
pub fn snap_to_square_perimeter_with_forbidden_sides(
    point_to_snap: Point<f32>,
    square_center: Point<f32>,
    square_side_length: f32,
    forbidden_sides: LocalBlockOccupancy,
) -> Point<f32> {
    let square_radius = square_side_length / 2.0;
    let relative_normalized_point = (point_to_snap - square_center) / square_radius;
    let mut candidate_rel_norm_snap_points = vec![];
    for i in 0..8 {
        let candidate_direction = adjacent_direction(i);
        let i = (candidate_direction.x() + 1) as usize;
        let j = (candidate_direction.y() + 1) as usize;
        let candidate_direction_is_valid = !forbidden_sides[i][j];
        if candidate_direction_is_valid {
            let rel_norm_snap_point = if candidate_direction.x() == 0 {
                p(
                    relative_normalized_point.x().clamp(-1.0, 1.0),
                    candidate_direction.y() as f32,
                )
            } else if candidate_direction.y() == 0 {
                p(
                    candidate_direction.x() as f32,
                    relative_normalized_point.y().clamp(-1.0, 1.0),
                )
            } else {
                floatify(candidate_direction)
            };
            candidate_rel_norm_snap_points.push(rel_norm_snap_point);
        }
    }
    let chosen_rel_norm_snap_point = *candidate_rel_norm_snap_points
        .iter()
        .min_by_key(|&&a| OrderedFloat(magnitude(relative_normalized_point - a)))
        .unwrap();
    chosen_rel_norm_snap_point * square_radius + square_center
}

pub fn snap_to_square_perimeter(
    point_to_snap: Point<f32>,
    square_center: Point<f32>,
    square_side_length: f32,
) -> Point<f32> {
    snap_to_square_perimeter_with_forbidden_sides(
        point_to_snap,
        square_center,
        square_side_length,
        [[false; 3]; 3],
    )
}

pub fn reflect_vector_over_axis(to_reflect: FPoint, reflection_axis: FPoint) -> FPoint {
    let axis_angle = angle_of_vector(reflection_axis);
    // 2d reflection matrix
    // A B
    // C D
    let a2 = 2.0 * axis_angle;
    let A = a2.cos();
    let B = a2.sin();
    let C = a2.sin();
    let D = -a2.cos();
    let (x, y) = to_reflect.x_y();
    p(x * A + y * B, x * C + y * D)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RADIUS_OF_EXACTLY_TOUCHING_ZONE;
    use assert2::assert;
    use ntest::assert_true;
    use std::f32::consts::PI;

    #[test]
    fn test_single_block_unit_squarecast__horizontal_hit() {
        let start = p(0.0, 0.0);
        let end = p(5.0, 0.0);
        let wall = p(2, 0);
        let result = single_block_unit_squarecast(start, end, wall);

        assert!(result.hit_something());
        assert!(points_nearly_equal(
            result.collider_pos,
            floatify(wall - p(1, 0))
        ));
        assert!(result.collision_normal.unwrap() == p(-1, 0));
    }

    #[test]
    fn test_unit_grid_square_overlap__one_square() {
        let point = p(57.0, -90.0);
        let squares = grid_squares_overlapped_by_floating_unit_square(point);
        assert!(squares.len() == 1);
        assert!(squares[0] == snap_to_grid(point));
    }
    #[test]
    fn test_unit_grid_square_overlap__two_squares_horizontal() {
        let point = p(0.5, 0.0);
        let squares = grid_squares_overlapped_by_floating_unit_square(point);
        assert!(squares.len() == 2);
        assert!(squares.contains(&p(0, 0)));
        assert!(squares.contains(&p(1, 0)));
    }

    #[test]
    fn test_unit_grid_square_overlap__two_squares_vertical() {
        let point = p(0.0, -0.1);
        let squares = grid_squares_overlapped_by_floating_unit_square(point);
        assert!(squares.len() == 2);
        assert!(squares.contains(&p(0, 0)));
        assert!(squares.contains(&p(0, -1)));
    }

    #[test]
    fn test_unit_grid_square_overlap__four_squares() {
        let point = p(5.9, -8.1);
        let squares = grid_squares_overlapped_by_floating_unit_square(point);
        assert!(squares.len() == 4);
        assert!(squares.contains(&p(5, -8)));
        assert!(squares.contains(&p(6, -8)));
        assert!(squares.contains(&p(5, -9)));
        assert!(squares.contains(&p(6, -9)));
    }
    #[test]
    fn test_unit_grid_square_barely_overlaps_four_squares() {
        let point = p(50.999, 80.0001);
        let squares = grid_squares_overlapped_by_floating_unit_square(point);
        assert!(squares.len() == 4);
        assert!(squares.contains(&p(50, 80)));
        assert!(squares.contains(&p(51, 80)));
        assert!(squares.contains(&p(50, 81)));
        assert!(squares.contains(&p(51, 81)));
    }

    #[test]
    fn test_offset_from_grid_rounds_to_zero() {
        assert!(offset_from_grid(p(9.0, -9.0)) == p(0.0, 0.0));
    }
    #[test]
    fn test_offset_from_grid_consistent_with_round_to_grid() {
        let mut p1 = p(0.0, 0.0);
        assert!(floatify(snap_to_grid(p1)) + offset_from_grid(p1) == p1);
        p1 = p(0.5, 0.5);
        assert!(floatify(snap_to_grid(p1)) + offset_from_grid(p1) == p1);
        p1 = p(-0.5, 0.5);
        assert!(floatify(snap_to_grid(p1)) + offset_from_grid(p1) == p1);
        p1 = p(-0.5, -0.5);
        assert!(floatify(snap_to_grid(p1)) + offset_from_grid(p1) == p1);
    }

    #[test]
    fn test_sign() {
        assert!(9.0.sign() == 1.0);
        assert!(0.1.sign() == 1.0);
        assert!(0.0.sign() == 0.0);
        assert!(-0.1.sign() == -1.0);
        assert!(-100.0.sign() == -1.0);

        assert!(9.sign() == 1);
        assert!(1.sign() == 1);
        assert!(0.sign() == 0);
        assert!(-1.sign() == -1);
        assert!(-100.sign() == -1);
    }

    #[test]
    fn test_vector_sign() {
        assert!(sign(p(9.0, -9.0)) == p(1.0, -1.0));
        assert!(sign(p(0.0, 0.0)) == p(0.0, 0.0));
        assert!(sign(p(-0.1, 0.1)) == p(-1.0, 1.0));
    }

    #[test]
    fn test_sign_of_offset_from_grid_rounds_to_zero() {
        assert!(sign(offset_from_grid(p(9.0, -9.0))) == p(0.0, 0.0));
    }
    #[test]
    fn test_snap_to_grid_at_zero() {
        assert!(snap_to_grid(p(0.0, 0.0)) == p(0, 0));
    }

    #[test]
    fn test_snap_to_grid_rounding_down_from_positive_x() {
        assert!(snap_to_grid(p(0.4, 0.0)) == p(0, 0));
    }

    #[test]
    fn test_snap_to_grid_rounding_up_diagonally() {
        assert!(snap_to_grid(p(0.9, 59.51)) == p(1, 60));
    }

    #[test]
    fn test_snap_to_grid_rounding_up_diagonally_in_the_negazone() {
        assert!(snap_to_grid(p(-0.9, -59.51)) == p(-1, -60));
    }

    #[test]
    fn test_single_block_unit_squarecast_head_on_horizontal() {
        let start_point = p(0.0, 0.0);
        let end_point = start_point + p(3.0, 0.0);
        let block_center = p(3, 0);
        assert!(points_nearly_equal(
            single_block_unit_squarecast(start_point, end_point, block_center).collider_pos,
            p(2.0, 0.0)
        ));
    }

    #[test]
    fn test_single_block_unit_squarecast__slightly_offset_from_vertical() {
        let start_point = p(0.3, 0.0);
        let end_point = start_point + p(0.0, 5.0);
        let block_center = p(0, 5);
        assert!(points_nearly_equal(
            single_block_unit_squarecast(start_point, end_point, block_center).collider_pos,
            p(start_point.x(), 4.0)
        ));
    }

    #[test]
    fn test_single_block_unit_squarecast__slightly_diagonalish() {
        let start_point = p(5.0, 0.0);
        let end_point = start_point + p(3.0, 3.0);
        let block_center = p(7, 1);
        assert!(points_nearly_equal(
            single_block_unit_squarecast(start_point, end_point, block_center).collider_pos,
            p(6.0, 1.0)
        ));
    }

    #[test]
    fn test_single_block_unit_squarecast__exactly_on_edge() {
        let start_point = p(0.0, 1.0);
        let end_point = start_point + p(10.0, 0.0);
        let block_center = p(4, 0);
        assert!(
            !single_block_unit_squarecast(start_point, end_point, block_center).hit_something()
        );
    }

    #[test]
    fn test_single_block_unit_squarecast__no_detection_moving_away_from_exact_touch() {
        let start_point = p(0.0, 0.0);
        let end_point = start_point + right_f() * 10.0;
        let block_center = p(-1, 0);
        assert!(
            !single_block_unit_squarecast(start_point, end_point, block_center).hit_something()
        );
    }

    #[test]
    fn test_single_block_linecast__exactly_on_top_edge() {
        let start_point = p(0.0, 0.5);
        let end_point = start_point + p(10.0, 0.0);
        let block_center = p(4, 0);
        assert!(!single_block_linecast(start_point, end_point, block_center).hit_something());
    }
    #[test]
    fn test_single_block_linecast__exactly_on_bottom_edge() {
        let start_point = p(0.0, -0.5);
        let end_point = start_point + p(10.0, 0.0);
        let block_center = p(4, 0);
        assert!(!single_block_linecast(start_point, end_point, block_center).hit_something());
    }

    #[test]
    fn test_single_block_linecast__no_detection_moving_away_from_exact_touch() {
        let block_center = p(0, 0);
        let start_point = p(0.5, 0.0);
        let end_point = start_point + right_f() * 10.0;
        assert!(!single_block_linecast(start_point, end_point, block_center).hit_something());
    }
    #[test]
    fn test_direction__nearly_zero() {
        let input = p(0.0, 1e-23);
        let output = direction(input);
        assert_eq!(magnitude(input), 0.0);
        assert!(!output.x().is_nan());
        assert!(!output.y().is_nan());
        assert!(!output.x().is_infinite());
        assert!(!output.y().is_infinite());
        assert_eq!(magnitude(output), 0.0);
    }

    #[test]
    fn test_orthogonal_direction_generation() {
        assert!(orthogonal_direction(0.0) == p(1.0, 0.0));
        assert!(orthogonal_direction(0) == p(1, 0));
        assert!(orthogonal_direction(1.0) == p(0.0, 1.0));
        assert!(orthogonal_direction(1) == p(0, 1));
        assert!(orthogonal_direction(2.0) == p(-1.0, 0.0));
        assert!(orthogonal_direction(2) == p(-1, 0));
        assert!(orthogonal_direction(3.0) == p(0.0, -1.0));
        assert!(orthogonal_direction(3) == p(0, -1));
        assert!(orthogonal_direction(4.0) == p(1.0, 0.0));
        assert!(orthogonal_direction(4) == p(1, 0));
    }

    #[test]
    fn test_chess_direction() {
        assert!(adjacent_direction(0) == p(1, 0));
        assert!(adjacent_direction(3.0) == p(-1.0, 1.0));
    }

    #[test]
    fn test_diagonal_direction() {
        assert!(diagonal_direction(0) == p(1, 1));
        assert!(diagonal_direction(2.0) == p(-1.0, -1.0));
    }

    #[test]
    fn test_projection() {
        // positive easy case
        assert!(project_a_onto_b(p(1.0, 0.0), p(5.0, 0.0)) == p(1.0, 0.0));
        assert!(project_a_onto_b(p(1.0, 0.0), p(0.0, 5.0)) == p(0.0, 0.0));
        // easy destination
        assert!(project_a_onto_b(p(1.0, 1.0), p(1.0, 0.0)) == p(1.0, 0.0));
        assert!(project_a_onto_b(p(6.0, 6.0), p(0.0, 1.0)) == p(0.0, 6.0));
        // more complex input
        assert!(project_a_onto_b(p(2.0, 6.0), p(0.0, 1.0)) == p(0.0, 6.0));
        // negative input
        assert!(project_a_onto_b(p(-6.0, 6.0), p(0.0, 1.0)) == p(0.0, 6.0));
        // zero
        assert!(project_a_onto_b(p(0.0, 0.0), p(0.0, 1.0)) == p(0.0, 0.0));
    }
    #[test]
    fn test_round_with_tie_break_to_inf() {
        assert!(round_vector_with_tie_break_toward_inf(p(0.0, 0.0)) == p(0, 0));
        assert!(round_vector_with_tie_break_toward_inf(p(1.0, 1.0)) == p(1, 1));
        assert!(round_vector_with_tie_break_toward_inf(p(-1.0, -1.0)) == p(-1, -1));
        assert!(round_vector_with_tie_break_toward_inf(p(0.1, 0.1)) == p(0, 0));
        assert!(round_vector_with_tie_break_toward_inf(p(-0.1, -0.1)) == p(0, 0));
        assert!(round_vector_with_tie_break_toward_inf(p(0.5, 0.5)) == p(1, 1));
        assert!(round_vector_with_tie_break_toward_inf(p(-0.5, -0.5)) == p(0, 0));
    }
    #[test]
    fn test_round_with_tie_break_to_neg_inf() {
        assert!(round_vector_with_tie_break_toward_neg_inf(p(0.0, 0.0)) == p(0, 0));
        assert!(round_vector_with_tie_break_toward_neg_inf(p(1.0, 1.0)) == p(1, 1));
        assert!(round_vector_with_tie_break_toward_neg_inf(p(-1.0, -1.0)) == p(-1, -1));
        assert!(round_vector_with_tie_break_toward_neg_inf(p(0.1, 0.1)) == p(0, 0));
        assert!(round_vector_with_tie_break_toward_neg_inf(p(-0.1, -0.1)) == p(0, 0));
        assert!(round_vector_with_tie_break_toward_neg_inf(p(0.5, 0.5)) == p(0, 0));
        assert!(round_vector_with_tie_break_toward_neg_inf(p(-0.5, -0.5)) == p(-1, -1));
    }

    #[test]
    fn test_decelerate_linearly_to_cap_under_cap() {
        assert!(decelerate_linearly_to_cap(1.0, 2.0, 1.0) == 1.0);
    }
    #[test]
    fn test_decelerate_linearly_to_cap_no_overshoot() {
        assert!(decelerate_linearly_to_cap(3.0, 2.0, 5.0) == 2.0);
    }
    #[test]
    fn test_decelerate_linearly_to_cap_partway_there() {
        assert!(decelerate_linearly_to_cap(4.0, 2.0, 1.0) == 3.0);
    }
    #[test]
    fn test_decelerate_linearly_to_cap_massive_overshoot() {
        assert!(decelerate_linearly_to_cap(4.0, 2.0, 100000.0) == 2.0);
    }
    #[test]
    fn test_decelerate_linearly_to_cap_negative_start() {
        assert!(decelerate_linearly_to_cap(-4.0, 2.0, 1.0) == -3.0);
    }
    #[test]
    fn test_decelerate_linearly_to_cap_stop_at_zero() {
        assert!(decelerate_linearly_to_cap(1.0, 0.0, 5.0) == 0.0);
    }

    #[test]
    fn test_accelerate_within_max_speed__within_bounds() {
        assert!(accelerate_within_max_speed(1.0, 1, 10.0, 1.0) == 2.0);
    }

    #[test]
    fn test_accelerate_within_max_speed__crossing_zero() {
        assert!(accelerate_within_max_speed(1.0, -1, 10.0, 3.0) == -2.0);
    }

    #[test]
    fn test_accelerate_within_max_speed__hitting_cap() {
        assert!(accelerate_within_max_speed(1.0, 1, 10.0, 10.0) == 10.0);
    }

    #[test]
    fn test_accelerate_within_max_speed__hitting_cap_across_zero() {
        assert!(accelerate_within_max_speed(1.0, -1, 10.0, 20.0) == -10.0);
    }

    #[test]
    fn test_accelerate_within_max_speed__stop_at_zero() {
        assert!(accelerate_within_max_speed(1.0, 0, 10.0, 5.0) == 0.0);
    }
    #[test]
    fn test_accelerate_within_max_speed__negative_slowing_down() {
        assert!(accelerate_within_max_speed(-1.0, 1, 10.0, 0.5) == -0.5);
    }

    #[test]
    fn test_accelerate_within_max_speed__negative_speeding_up() {
        assert!(accelerate_within_max_speed(-1.0, -1, 10.0, 0.5) == -1.5);
    }

    #[test]
    fn test_accelerate_within_max_speed__stopping_from_above_max() {
        assert!(accelerate_within_max_speed(10.0, 0, 5.0, 1.0) == 9.0);
    }
    #[test]
    fn test_accelerate_within_max_speed__slowing_from_above_max() {
        assert!(accelerate_within_max_speed(10.0, -1, 5.0, 1.0) == 9.0);
    }
    #[test]
    fn test_accelerate_within_max_speed__failing_to_go_fast() {
        assert!(accelerate_within_max_speed(10.0, 1, 5.0, 1.0) == 10.0);
    }
    #[test]
    fn test_accelerate_within_max_speed__stopping_from_below_neg_max() {
        assert!(accelerate_within_max_speed(-10.0, 0, 5.0, 1.0) == -9.0);
    }
    #[test]
    fn test_accelerate_within_max_speed__slowing_from_below_neg_max() {
        assert!(accelerate_within_max_speed(-10.0, 1, 5.0, 1.0) == -9.0);
    }

    #[test]
    fn test_compensate_for_vertical_stretch() {
        assert!(world_space_to_grid_space(p(5.0, 2.0), 2.0) == p(5.0, 1.0));
    }

    #[test]
    fn test_lerp_halfway() {
        assert_relative_eq!(lerp(0.0, 10.0, 0.5), 5.0);
    }
    #[test]
    fn test_lerp_low_bound() {
        assert_relative_eq!(lerp(0.0, 10.0, 0.0), 0.0);
    }
    #[test]
    fn test_lerp_high_bound() {
        assert_relative_eq!(lerp(0.0, 10.0, 1.0), 10.0);
    }
    #[test]
    fn test_lerp_past_low_negative_bound() {
        assert_relative_eq!(lerp(-5.0, 5.0, -0.5), -10.0);
    }
    #[test]
    fn test_lerp_within_reversed_bounds() {
        assert_relative_eq!(lerp(100.0, 0.0, 0.2), 80.0);
    }

    #[test]
    fn test_inverse_lerp_undoes_lerp() {
        let a = 5.0;
        let b = 100.0;

        let t = 3.445;
        assert_relative_eq!(inverse_lerp(a, b, lerp(a, b, t)), t);

        let t = 0.384;
        assert_relative_eq!(inverse_lerp(a, b, lerp(a, b, t)), t);

        let t = -25.0;
        assert_relative_eq!(inverse_lerp(a, b, lerp(a, b, t)), t);
    }

    #[test]
    fn test_square_exactly_touching_square_below() {
        assert!(floating_square_orthogonally_touching_fixed_square(
            p(1.0, 0.0),
            p(1, -1)
        ));
    }
    #[test]
    fn test_square_exactly_touching_square_below_with_small_tolerance() {
        assert!(floating_square_orthogonally_touching_fixed_square(
            p(1.0, 0.0 + RADIUS_OF_EXACTLY_TOUCHING_ZONE / 2.0),
            p(1, -1)
        ));
    }
    #[test]
    fn test_square_exactly_touching_square_below_with_small_horizontal_shift() {
        assert!(floating_square_orthogonally_touching_fixed_square(
            p(1.2, 0.0),
            p(1, -1)
        ));
    }
    #[test]
    fn test_square_exactly_touching_square_below_with_larger_horizontal_shift() {
        assert!(floating_square_orthogonally_touching_fixed_square(
            p(1.9, 0.0),
            p(1, -1)
        ));
    }
    #[test]
    fn test_square_not_exactly_touching_square_below_with_way_too_big_horizontal_shift() {
        assert!(!floating_square_orthogonally_touching_fixed_square(
            p(19.0, 0.0),
            p(1, -1)
        ));
    }
    #[test]
    fn test_square_not_exactly_touching_square_below_with_slight_gap() {
        assert!(!floating_square_orthogonally_touching_fixed_square(
            p(1.0, 0.0001),
            p(1, -1)
        ));
    }
    #[test]
    fn test_square_not_exactly_touching_square_below_with_slight_overlap() {
        assert!(!floating_square_orthogonally_touching_fixed_square(
            p(1.0, -0.0001),
            p(1, -1)
        ));
    }
    #[test]
    fn test_square_not_exactly_touching_square_on_perfect_diagonal() {
        assert!(!floating_square_orthogonally_touching_fixed_square(
            p(0.0, 0.0),
            p(1, -1)
        ));
    }

    #[test]
    fn test_lerp_2d_quarter_to_end() {
        let y = 5.0;
        assert!(lerp_2d(p(4.0, y), p(8.0, y), 0.75) == p(7.0, y));
    }

    #[test]
    fn test_lerp_2d_at_end_of_diagonal() {
        assert!(lerp_2d(p(1.0, 8.0), p(5.0, -7.0), 1.0) == p(5.0, -7.0));
    }
    #[test]
    fn test_lerp_2d_at_start_of_diagonal() {
        assert!(lerp_2d(p(-1.0, -8.0), p(5.0, 7777.0), 0.0) == p(-1.0, -8.0));
    }

    #[test]
    fn test_line_of_points__no_intermediates() {
        assert!(points_in_line_with_max_gap(p(1.0, 0.0), p(2.0, 0.0), 1.0).len() == 2);
    }

    #[test]
    fn test_line_of_points__vertical() {
        let y = 6.9;
        let density = 9.0;
        assert!(
            points_in_line_with_max_gap(p(0.0, 0.0), p(0.0, y), density).len()
                == ((density * y).ceil() as usize)
        );
    }
    #[test]
    fn test_line_of_points__exact_endpoints() {
        let start = p(0.12309, 4.234); //arbitrary
        let end = p(-0.12309, 45.28374); //arbitrary
        let density = 1.1234;
        let points = points_in_line_with_max_gap(start, end, density);
        assert!(*points.first().unwrap() == start);
        assert!(*points.last().unwrap() == end);
    }
    #[test]
    fn test_line_of_points__symmetric() {
        let start = p(0.1209, 1.234); //arbitrary
        let end = p(1.12309, 1.28374); //arbitrary
        let density = 2.0;
        let points = points_in_line_with_max_gap(start, end, density);
        assert!(points.len() == 3);
        let midpoint = points[1];
        let d1 = magnitude(start - midpoint);
        let d2 = magnitude(end - midpoint);

        assert_relative_eq!(d1, d2);
    }
    #[test]
    fn test_rotated() {
        assert!(abs_diff_eq!(
            rotated_degrees(p(1.0, 0.0), 90.0).x(),
            0.0,
            epsilon = 0.000001
        ));
        assert!(abs_diff_eq!(
            rotated_degrees(p(1.0, 0.0), 90.0).y(),
            1.0,
            epsilon = 0.000001
        ));

        assert!(abs_diff_eq!(
            rotated_degrees(p(0.0, 1.0), 90.0).x(),
            -1.0,
            epsilon = 0.000001
        ));
        assert!(abs_diff_eq!(
            rotated_degrees(p(0.0, 1.0), 90.0).y(),
            0.0,
            epsilon = 0.000001
        ));

        assert!(abs_diff_eq!(
            rotated_degrees(p(-1.0, 0.0), 90.0).x(),
            0.0,
            epsilon = 0.000001
        ));
        assert!(abs_diff_eq!(
            rotated_degrees(p(-1.0, 0.0), 90.0).y(),
            -1.0,
            epsilon = 0.000001
        ));

        assert!(abs_diff_eq!(
            rotated_degrees(p(0.0, -1.0), 90.0).x(),
            1.0,
            epsilon = 0.000001
        ));
        assert!(abs_diff_eq!(
            rotated_degrees(p(0.0, -1.0), 90.0).y(),
            0.0,
            epsilon = 0.000001
        ));
    }
    #[test]
    fn test_time_synchronized_points_on_line__simple_horizontal_no_rounding() {
        let points = time_synchronized_points_on_line(p(0.0, 0.0), p(5.0, 0.0), -0.1, 9.9, 1.0);
        assert!(points.len() == 10);
        for p in points {
            assert!(p.y() == 0.0);
            assert!(p.x() <= 5.0);
            assert!(p.x() >= 0.0);
        }
    }
    #[test]
    fn test_time_synchronized_points_on_line__include_start_time_but_not_end_time() {
        let start_point = p(0.0, 0.0);
        let points = time_synchronized_points_on_line(start_point, p(1.0, 1.0), 0.0, 1.0, 1.0);
        assert!(points.len() == 1);
        assert!(points[0] == start_point);
    }
    #[test]
    fn test_time_synchronized_points_on_line__can_be_empty() {
        let start_point = p(0.0, 0.0);
        let points = time_synchronized_points_on_line(start_point, p(1.0, 1.0), 0.1, 1.0, 1.0);
        assert!(points.is_empty());
    }
    #[test]
    fn test_time_synchronized_points_on_line__correct_interpolation() {
        let start_point = p(0.0, 0.0);
        let end_point = p(2.0, 1.0);
        let points = time_synchronized_points_on_line(start_point, end_point, 0.25, 1.25, 1.0);
        assert!(points.len() == 1);
        assert!(points[0] == p(1.5, 0.75));
    }
    #[test]
    fn test_grid_square_overlap__really_small_square() {
        let point = p(1.1, 1.9);
        let squares = grid_squares_overlapped_by_floating_square(point, 0.2);
        assert!(squares.len() == 1);
        assert!(squares.contains(&p(1, 2)));
    }
    #[test]
    fn test_grid_square_overlap__really_small_square_on_corner() {
        let point = p(1.5, 1.5);
        let squares = grid_squares_overlapped_by_floating_square(point, 0.2);
        assert!(squares.len() == 4);
    }
    #[test]
    fn test_grid_square_overlap__9_squares_when_centered() {
        let point = p(5.0, 5.0);
        let squares = grid_squares_overlapped_by_floating_square(point, 1.1);
        assert!(squares.len() == 9);
    }
    #[test]
    fn test_grid_square_overlap__exactly_overlapping_4_and_touching_12_more() {
        let point = p(5.5, 5.5);
        let squares = grid_squares_overlapped_by_floating_square(point, 2.0);
        assert!(squares.len() == 4);
        assert!(squares.contains(&p(5, 5)));
        assert!(squares.contains(&p(6, 5)));
        assert!(squares.contains(&p(5, 6)));
        assert!(squares.contains(&p(6, 6)));
    }

    #[test]
    fn test_point_inside_square() {
        assert!(point_inside_grid_square(p(0.49, 0.6), p(0, 0)) == false);
        assert!(point_inside_grid_square(p(0.49, 0.4), p(0, 0)) == true);
        assert!(point_inside_grid_square(p(0.49, 0.6), p(0, 1)) == true);
        assert!(point_inside_grid_square(p(5.0, 5.1), p(5, 5)) == true);
        assert!(point_inside_grid_square(p(5.0, 8.1), p(5, 5)) == false);
    }

    #[test]
    fn test_point_exactly_touching_square() {
        assert!(point_exactly_touching_unit_square(p(0.5, 0.5), p(0, 0)) == true);
        assert!(point_exactly_touching_unit_square(p(0.5, 0.5), p(1, 0)) == true);
        assert!(point_exactly_touching_unit_square(p(0.5, 0.5), p(1, 1)) == true);
        assert!(point_exactly_touching_unit_square(p(0.5, 0.5), p(0, 1)) == true);

        assert!(point_exactly_touching_unit_square(p(0.5001, 0.5), p(0, 0)) == false);
        assert!(point_exactly_touching_unit_square(p(0.5001, 0.5), p(1, 0)) == true);
        assert!(point_exactly_touching_unit_square(p(0.5001, 0.5), p(0, 1)) == false);
        assert!(point_exactly_touching_unit_square(p(0.5001, 0.5), p(1, 1)) == true);
    }
    #[test]
    fn test_point_exactly_touching_square_with_small_tolerance() {
        assert!(
            point_exactly_touching_unit_square(
                p(0.5 + RADIUS_OF_EXACTLY_TOUCHING_ZONE / 2.0, 0.0),
                p(0, 0)
            ) == true
        );
        assert!(
            point_exactly_touching_unit_square(
                p(0.5 + 3.0 * RADIUS_OF_EXACTLY_TOUCHING_ZONE / 2.0, 0.0),
                p(0, 0)
            ) == false
        );
    }
    #[test]
    fn test_point_not_inside_square_if_exactly_touching_it_within_tolerances() {
        let test_point_touching_square = p(0.5 + RADIUS_OF_EXACTLY_TOUCHING_ZONE / 2.0, 0.0);
        let test_point_in_square = p(0.5 - 3.0 * RADIUS_OF_EXACTLY_TOUCHING_ZONE / 2.0, 0.0);
        let test_square = p(0, 0);

        assert!(
            point_exactly_touching_unit_square(test_point_touching_square, test_square)
                != point_inside_grid_square(test_point_touching_square, test_square)
        );
        assert!(
            point_exactly_touching_unit_square(test_point_in_square, test_square)
                != point_inside_grid_square(test_point_in_square, test_square)
        );
    }
    #[test]
    fn test_exactly_touching_is_not_enough_to_count_as_overlapping() {
        assert!(
            grid_squares_overlapped_by_floating_square(
                p(0.0, 0.0),
                1.0 + RADIUS_OF_EXACTLY_TOUCHING_ZONE / 2.0
            )
            .len()
                == 1
        );
    }

    #[test]
    fn test_int_cos() {
        assert!(int_cos(-1) == 0);
        assert!(int_cos(0) == 1);
        assert!(int_cos(1) == 0);
        assert!(int_cos(2) == -1);
        assert!(int_cos(3) == 0);
        assert!(int_cos(4) == 1);
    }
    #[test]
    fn test_int_sin() {
        assert!(int_sin(-1) == -1);
        assert!(int_sin(0) == 0);
        assert!(int_sin(1) == 1);
        assert!(int_sin(2) == 0);
        assert!(int_sin(3) == -1);
        assert!(int_sin(4) == 0);
    }
    #[test]
    fn test_quarter_turns() {
        assert!(quarter_turns_counter_clockwise(p(1.0, 2.0), 1) == p(-2.0, 1.0));
        assert!(quarter_turns_counter_clockwise(p(1, 2), 1) == p(-2, 1));
    }
    #[test]
    fn test_snap_to_square_perimeter() {
        assert!(snap_to_square_perimeter(p(1.1, 1.0), p(0.5, 0.5), 1.0) == p(1.0, 1.0));
        assert!(snap_to_square_perimeter(p(1.1, 0.7), p(0.5, 0.5), 1.0) == p(1.0, 0.7));
        assert!(snap_to_square_perimeter(p(1.000001, 1.0), p(0.5, 0.5), 1.0) == p(1.0, 1.0));
        assert!(snap_to_square_perimeter(p(0.5, 1.0), p(0.0, 0.0), 1.0) == p(0.5, 0.5));
        assert!(snap_to_square_perimeter(p(6.0, 5.7), p(5.0, 5.0), 1.0) == p(5.5, 5.5));
    }
    #[test]
    fn test_snap_to_square_perimeter_with_forbidden_sides() {
        let mut top_filled: LocalBlockOccupancy = [[false; 3]; 3];
        top_filled[1][2] = true;

        let square = p(1, 1);
        let square_side_length = 1.0;
        let start_point = p(0.5001, 1.5);
        let good_end_point = p(0.5, 1.5);

        assert!(
            snap_to_square_perimeter(start_point, floatify(square), square_side_length)
                == start_point
        );
        assert!(
            snap_to_square_perimeter_with_forbidden_sides(
                start_point,
                floatify(square),
                square_side_length,
                top_filled
            ) == good_end_point
        );
    }

    #[test]
    fn test_single_block_squarecast_filled_cracks__no_normal_within_wall() {
        // data from observed failure
        let start_point = p(8.5, 12.5);
        let end_point = p(8.199999, 12.5);
        let grid_square_center = p(8, 13);
        let adjacent_squares_occupied = [
            [false, false, false],
            [true, true, true],
            [false, false, false],
        ]; // vertical wall

        let collision = single_block_linecast_with_filled_cracks(
            start_point,
            end_point,
            grid_square_center,
            adjacent_squares_occupied,
        );
        assert!(collision.collision_normal.unwrap().x() != 0);
        assert!(collision.collision_normal.unwrap().y() == 0);
    }
    #[test]
    fn test_single_block_squarecast_filled_cracks__do_not_expand_corner_when_only_surrounded_orthogonally(
    ) {
        let grid_square_center = p(0, 0);
        let start_point = floatify(grid_square_center) + p(-10.0, 10.0);
        let top_left_corner_of_block = floatify(grid_square_center) + p(-0.5, 0.5);
        let endpoint_offset_for_test =
            (right_f() + down_f()) * RADIUS_OF_EXACTLY_TOUCHING_ZONE / 2.0;
        let end_point = top_left_corner_of_block + endpoint_offset_for_test;
        let adjacent_squares_occupied = visible_xy_to_actual_xy([
            [false, true, true],
            [true, true, true],
            [false, false, false],
        ]); // orthogonally surrounded
        let maybe_collision = single_block_linecast_with_filled_cracks(
            start_point,
            end_point,
            grid_square_center,
            adjacent_squares_occupied,
        );
        //dbg!(start_point, end_point, &maybe_collision);
        assert!(!maybe_collision.hit_something());
    }

    #[test]
    fn test_angle_between() {
        assert!(nearly_equal(angle_between(right_f(), right_f()), 0.0));
        assert!(nearly_equal(angle_between(right_f(), up_f()), PI / 2.0));
        assert!(nearly_equal(angle_between(down_f(), up_f()), PI));
        assert!(nearly_equal(
            angle_between(down_f(), p(-10.0, -10.0)),
            PI / 4.0
        ));
    }
    #[test]
    fn test_rotate_angle_towards__simple_positive() {
        assert!(nearly_equal(rotate_angle_towards(0.0, 1.0, 0.1), 0.1));
    }
    #[test]
    fn test_rotate_angle_towards__no_overshoot_positive() {
        assert!(nearly_equal(rotate_angle_towards(0.0, 1.0, 10.0), 1.0));
    }
    #[test]
    fn test_rotate_angle_towards__simple_negative() {
        assert!(nearly_equal(rotate_angle_towards(0.0, -1.0, 0.1), -0.1));
    }
    #[test]
    fn test_rotate_angle_towards__do_something_for_180_degree_case() {
        assert!(!nearly_equal(rotate_angle_towards(0.0, PI, 0.1), PI));
    }
    #[test]
    fn test_rotate_angle_towards__modulo_two_pi() {
        assert!(nearly_equal(
            to_standard_angle(rotate_angle_towards(4.0 * PI, 1.0, 0.1)),
            0.1
        ));
    }
    #[test]
    fn test_rotate_angle_towards__across_branch_point() {
        assert!(nearly_equal(
            rotate_angle_towards(0.1, TAU - 0.1, 0.01),
            0.09
        ));
    }
    #[test]
    fn test_rotate_angle_towards__across_branch_point_reversed() {
        assert!(nearly_equal(
            rotate_angle_towards(TAU - 0.1, 0.1, 0.01),
            TAU - 0.09
        ));
    }

    #[test]
    fn test_flip_x() {
        assert!(flip_x(p(5.0, 8.0)) == p(-5.0, 8.0));
        assert!(flip_x(p(-5.0, 8.0)) == p(5.0, 8.0));
    }

    #[test]
    fn test_flip_y() {
        assert!(flip_y(p(5.0, 8.0)) == p(5.0, -8.0));
        assert!(flip_y(p(5.0, -8.0)) == p(5.0, 8.0));
    }

    #[test]
    fn test_when_linear_motion_hits_circle__simple_test() {
        assert!(
            when_linear_motion_hits_circle_from_inside(right_f() * 5.0, right_f() * 1.0, 7.0)
                == Some(2.0)
        );
        assert!(
            when_linear_motion_hits_circle_from_inside(right_f() * 5.0, left_f() * 1.0, 4.0)
                == Some(1.0)
        );
        assert!(
            when_linear_motion_hits_circle_from_inside(up_f() * 5.0, down_f() * 1.0, 7.0)
                == Some(12.0)
        );
        assert!(
            when_linear_motion_hits_circle_from_inside(right_f() * 5.0, right_f() * 1.0, 4.0)
                == None
        );
    }
    #[test]
    fn test_when_linear_motion_hits_circle__zeros_test() {
        assert!(when_linear_motion_hits_circle_from_inside(zero_f(), zero_f(), 4.0) == None);
        assert!(
            when_linear_motion_hits_circle_from_inside(right_f() * 25.0, zero_f(), 4.0) == None
        );
        assert!(when_linear_motion_hits_circle_from_inside(zero_f(), right_f(), 0.0) == Some(0.0));
        assert!(when_linear_motion_hits_circle_from_inside(zero_f(), zero_f(), 0.0) == None);
    }
    #[test]
    fn test_when_linear_motion_hits_circle__fancy_test() {
        let x0 = p(5.345, 1.567); // random
        let v = direction(p(1.398, 8.4837));
        let t = 4.567;
        let endpoint = x0 + v * t;
        let end_radius = magnitude(endpoint);

        let calculated_time =
            when_linear_motion_hits_circle_from_inside(x0, v, end_radius).unwrap();
        assert!(nearly_equal(t, calculated_time));
    }
    #[test]
    fn test_when_linear_motion_hits_circle__start_on_circle_moving_out() {
        let r = 5.0;
        assert!(
            when_linear_motion_hits_circle_from_inside(
                right_f() * r,
                right_f() * 1.0 + up_f() * 0.5,
                r
            ) == Some(0.0)
        );
    }
    #[test]
    fn test_when_linear_motion_hits_circle__start_on_circle_moving_in() {
        let r = 5.0;
        let t = when_linear_motion_hits_circle_from_inside(
            right_f() * r,
            left_f() * 1.0 + up_f() * 0.5,
            r,
        );
        assert!(t.is_some());
        assert!(t.unwrap() > 0.0);
    }

    #[test]
    fn test_extrapolate_kinematics__accelerate_right() {
        let start_state = KinematicState {
            pos: zero_f(),
            vel: right_f() * 1.0,
            accel: right_f() * 1.0,
        };
        let end_state = start_state.extrapolated(1.0);
        assert!(end_state.vel == start_state.vel + right_f() * 1.0);
    }
    #[test]
    fn test_extrapolate_kinematics_with_speed_cap__do_not_exceed_cap() {
        let start_state = KinematicState {
            pos: zero_f(),
            vel: right_f() * 1.0,
            accel: right_f() * 1.0,
        };
        let end_state = start_state.extrapolated_with_speed_cap(1.0, 1.0);
        assert!(end_state.vel == start_state.vel);
    }
    #[test]
    fn test_extrapolate_kinematics_with_speed_cap__can_drop_below_cap_from_above_cap() {
        let speed_cap = 1.0;
        let start_state = KinematicState {
            pos: zero_f(),
            vel: right_f() * 1.1,
            accel: left_f() * 1.0,
        };
        let end_state = start_state.extrapolated_with_speed_cap(1.0, speed_cap);
        assert!(magnitude(end_state.vel) < speed_cap);
    }
    #[test]
    fn test_extrapolate_kinematics_with_speed_cap__can_drop_below_cap_from_at_cap() {
        let speed_cap = 1.0;
        let start_state = KinematicState {
            pos: zero_f(),
            vel: right_f() * 1.0,
            accel: left_f() * 0.05,
        };
        let end_state = start_state.extrapolated_with_speed_cap(1.0, speed_cap);
        assert!(magnitude(end_state.vel) < speed_cap);
    }
    #[test]
    fn test_direction_of_zero_is_zero() {
        assert!(direction(zero_f()) == zero_f());
    }
    #[test]
    fn test_extrapolated_delta_kinematics_with_speed_cap__reasonable_dv() {
        let dt = 0.997;
        let start_state = KinematicState {
            pos: p(15.0, 10.0),
            vel: zero_f(),
            accel: down_f() * 0.1,
        };
        let speed_cap = 0.7;
        let diff = start_state.extrapolated_delta_with_speed_cap(dt, speed_cap);
        assert!(magnitude(diff.vel) < magnitude(start_state.accel) * 1.5);
    }
    #[test]
    fn test_extrapolate_kinematics_with_full_stop_at_slowest__upwards_arc() {
        let start_state = KinematicState {
            pos: zero_f(),
            vel: right_f() * 3.0 + up_f() * 1.0,
            accel: down_f() * 1.0,
        };
        assert!(start_state.extrapolated_with_full_stop_at_slowest(2.0).vel == zero_f());
    }
    #[test]
    fn test_extrapolate_kinematics_with_full_stop_at_slowest__past_slowest() {
        let start_state = KinematicState {
            pos: zero_f(),
            vel: right_f(),
            accel: right_f(),
        };
        assert!(start_state.extrapolated_with_full_stop_at_slowest(2.0).vel != zero_f());
    }

    #[ignore] // Not sure about this one
    #[test]
    fn test_extrapolate_kinematics_with_full_stop_at_slowest__no_stop_at_start() {
        let start_state = KinematicState {
            pos: zero_f(),
            vel: zero_f(),
            accel: right_f(),
        };
        assert!(start_state.extrapolated_with_full_stop_at_slowest(2.0).vel != zero_f());
    }

    #[test]
    fn test_extrapolate_kinematics_with_full_stop_at_slowest__before_slowest() {
        let start_state = KinematicState {
            pos: zero_f(),
            vel: left_f() * 100.0,
            accel: right_f(),
        };
        let dt = 2.0;
        assert!(
            start_state.extrapolated_with_full_stop_at_slowest(dt) == start_state.extrapolated(dt)
        );
    }
    #[test]
    fn test_extrapolate_kinematics_with_full_stop_at_slowest__constant_non_zero_speed_means_no_stop(
    ) {
        let start_state = KinematicState {
            pos: zero_f(),
            vel: left_f() * 100.0,
            accel: zero_f(),
        };
        let dt = 2.0;
        assert!(
            start_state.extrapolated_with_full_stop_at_slowest(dt) == start_state.extrapolated(dt)
        );
    }
    #[test]
    fn test_extrapolate_kinematics_with_full_stop_at_slowest__zero_to_zero() {
        let start_state = KinematicState {
            pos: zero_f(),
            vel: zero_f(),
            accel: zero_f(),
        };
        assert!(
            start_state
                .extrapolated_with_full_stop_at_slowest(500.0)
                .vel
                == zero_f()
        );
    }

    #[test]
    fn test_extrapolate_kinematics_with_full_stop_at_slowest__straight_turnaround() {
        let start_state = KinematicState {
            pos: zero_f(),
            vel: up_f() * 1.5,
            accel: down_f() * 1.0,
        };
        assert!(start_state.extrapolated_with_full_stop_at_slowest(2.0).vel == zero_f());
    }

    #[test]
    fn test_extrapolate_kinematics_with_full_stop_at_slowest__straight_turnaround_with_strange_direction(
    ) {
        let dir = normalized(p(-1.3, 0.03645));
        let start_state = KinematicState {
            pos: zero_f(),
            vel: dir * 5.393,
            accel: -dir * 2.1,
        };
        assert!(
            start_state
                .extrapolated_with_full_stop_at_slowest(100.0)
                .vel
                == zero_f()
        );
    }

    #[test]
    fn test_reflection() {
        assert!(points_nearly_equal(
            reflect_vector_over_axis(left_f(), up_f()),
            right_f()
        ));
        assert!(points_nearly_equal(
            reflect_vector_over_axis(left_f() * 123.0, down_f() * 0.3),
            right_f() * 123.0
        ));
        assert!(points_nearly_equal(
            reflect_vector_over_axis(p(201.0, 3.5), right_f()),
            p(201.0, -3.5)
        ));
        assert!(points_nearly_equal(
            reflect_vector_over_axis(p(0.0, 5.0), up_right_f()),
            p(5.0, 0.0)
        ));
    }
}
