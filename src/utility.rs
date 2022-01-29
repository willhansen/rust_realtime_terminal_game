extern crate geo;

use crate::num::traits::Pow;
use geo::algorithm::euclidean_distance::EuclideanDistance;
use geo::algorithm::line_intersection::{line_intersection, LineIntersection};
use geo::{point, CoordNum, Point};

pub fn p<T: 'static>(x: T, y: T) -> Point<T>
where
    T: CoordNum,
{
    return point!(x: x, y: y);
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct MovecastCollision {
    pub pos: Point<f32>,
    pub normal: Point<i32>,
}

pub fn single_block_movecast(
    start_point: Point<f32>,
    end_point: Point<f32>,
    grid_square_center: Point<i32>,
) -> Option<MovecastCollision>
where
{
    // formulates the problem as a point crossing the boundary of an r=1 square
    let movement_line = geo::Line::new(start_point, end_point);
    //println!("movement_line: {:?}", movement_line);
    let expanded_corner_offsets = vec![p(1.0, 1.0), p(-1.0, 1.0), p(-1.0, -1.0), p(1.0, -1.0)];
    let expanded_square_corners: Vec<Point<f32>> = expanded_corner_offsets
        .iter()
        .map(|&rel_p| floatify(grid_square_center) + rel_p)
        .collect();
    let expanded_square_edges = vec![
        geo::Line::new(expanded_square_corners[0], expanded_square_corners[1]),
        geo::Line::new(expanded_square_corners[1], expanded_square_corners[2]),
        geo::Line::new(expanded_square_corners[2], expanded_square_corners[3]),
        geo::Line::new(expanded_square_corners[3], expanded_square_corners[0]),
    ];
    //println!("expanded_edges: {:?}", expanded_square_edges);
    let mut candidate_edge_intersections = Vec::<Point<f32>>::new();
    for edge in expanded_square_edges {
        if let Some(LineIntersection::SinglePoint {
            intersection: coord,
            is_proper: _,
        }) = line_intersection(movement_line, edge)
        {
            candidate_edge_intersections.push(coord.into());
        }
    }
    if candidate_edge_intersections.is_empty() {
        return None;
    }

    // four intersections with extended walls of stationary square
    candidate_edge_intersections.sort_by(|a, b| {
        start_point
            .euclidean_distance(a)
            .partial_cmp(&start_point.euclidean_distance(b))
            .unwrap()
    });
    //println!("intersections: {:?}", candidate_edge_intersections);
    let collision_point = candidate_edge_intersections[0];
    //println!("collision_point: {:?}", collision_point);
    //println!("rounded_dir_number: {:?}", round_to_direction_number( collision_point - floatify(grid_square_center)));
    let collision_normal = e(round_to_direction_number(
        collision_point - floatify(grid_square_center),
    ));
    return Some(MovecastCollision {
        pos: collision_point,
        normal: collision_normal,
    });
}

pub fn e<T: CoordNum + num::Signed + std::fmt::Display>(dir_num: T) -> Point<T> {
    let dir_num_int = dir_num.to_i32().unwrap() % 4;
    match dir_num_int {
        0 => Point::<T>::new(T::one(), T::zero()),
        1 => Point::<T>::new(T::zero(), T::one()),
        2 => Point::<T>::new(-T::one(), T::zero()),
        3 => Point::<T>::new(T::zero(), -T::one()),
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
    let mut output = Vec::<Point<i32>>::new();
    let offset_direction = round(sign(offset_from_grid(pos)));
    // each non-zero offset axis implies one more square.  Both implies three
    for i in 0..3 {
        for j in 0..3 {
            let candidate_square_pos = p(i as i32 - 1, j as i32 - 1);
            if (candidate_square_pos.x() == offset_direction.x() || candidate_square_pos.x() == 0)
                && (candidate_square_pos.y() == offset_direction.y()
                    || candidate_square_pos.y() == 0)
            {
                output.push(snap_to_grid(pos) + candidate_square_pos);
            }
        }
    }
    return output;
}

pub fn snap_to_grid(world_pos: Point<f32>) -> Point<i32> {
    return round(world_pos);
}
pub fn offset_from_grid(world_pos: Point<f32>) -> Point<f32> {
    return world_pos - floatify(snap_to_grid(world_pos));
}

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
pub fn round_with_tie_break_toward_inf(x: f32) -> i32 {
    if (x - x.round()).abs() == 0.5 {
        return (x + 0.1).round() as i32;
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
pub fn direction(vec: Point<f32>) -> Point<f32> {
    return vec / magnitude(vec);
}

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
        if *self == T::zero() {
            return T::zero();
        } else if self.is_negative() {
            return -T::one();
        } else {
            return T::one();
        }
    }
}

pub fn project(v1: Point<f32>, v2: Point<f32>) -> Point<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;

    #[test]
    fn test_single_block_movecast_horizontal_hit() {
        let start = p(0.0, 0.0);
        let end = p(5.0, 0.0);
        let wall = p(2, 0);
        let result = single_block_movecast(start, end, wall);

        assert!(result != None);
        assert!(result.unwrap().pos == floatify(wall - p(1, 0)));
        assert!(result.unwrap().normal == p(-1, 0));
    }

    #[test]
    fn grid_square_overlap_one_square() {
        let point = p(57.0, -90.0);
        let squares = grid_squares_overlapped_by_floating_unit_square(point);
        assert!(squares.len() == 1);
        assert!(squares[0] == snap_to_grid(point));
    }
    #[test]
    fn grid_square_overlap_two_squares_horizontal() {
        let point = p(0.5, 0.0);
        let squares = grid_squares_overlapped_by_floating_unit_square(point);
        assert!(squares.len() == 2);
        assert!(squares.contains(&p(0, 0)));
        assert!(squares.contains(&p(1, 0)));
    }

    #[test]
    fn grid_square_overlap_two_squares_vertical() {
        let point = p(0.0, -0.1);
        let squares = grid_squares_overlapped_by_floating_unit_square(point);
        assert!(squares.len() == 2);
        assert!(squares.contains(&p(0, 0)));
        assert!(squares.contains(&p(0, -1)));
    }

    #[test]
    fn grid_square_overlap_four_squares() {
        let point = p(5.9, -8.1);
        let squares = grid_squares_overlapped_by_floating_unit_square(point);
        assert!(squares.len() == 4);
        assert!(squares.contains(&p(5, -8)));
        assert!(squares.contains(&p(6, -8)));
        assert!(squares.contains(&p(5, -9)));
        assert!(squares.contains(&p(6, -9)));
    }
    #[test]
    fn grid_square_barely_overlaps_four_squares() {
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
    fn test_collision_point_head_on_horizontal() {
        let start_point = p(0.0, 0.0);
        let end_point = start_point + p(3.0, 0.0);
        let block_center = p(3, 0);
        assert!(
            single_block_movecast(start_point, end_point, block_center)
                .unwrap()
                .pos
                == p(2.0, 0.0)
        );
    }

    #[test]
    fn test_collision_point_head_slightly_offset_from_vertical() {
        let start_point = p(0.3, 0.0);
        let end_point = start_point + p(0.0, 5.0);
        let block_center = p(0, 5);
        assert!(
            single_block_movecast(start_point, end_point, block_center)
                .unwrap()
                .pos
                == p(start_point.x(), 4.0)
        );
    }

    #[test]
    fn test_collision_point_slightly_diagonalish() {
        let start_point = p(5.0, 0.0);
        let end_point = start_point + p(3.0, 3.0);
        let block_center = p(7, 1);
        assert!(
            single_block_movecast(start_point, end_point, block_center)
                .unwrap()
                .pos
                == p(6.0, 1.0)
        );
    }

    #[test]
    fn test_orthogonal_direction_generation() {
        assert!(e(0.0) == p(1.0, 0.0));
        assert!(e(0) == p(1, 0));
        assert!(e(1.0) == p(0.0, 1.0));
        assert!(e(1) == p(0, 1));
        assert!(e(2.0) == p(-1.0, 0.0));
        assert!(e(2) == p(-1, 0));
        assert!(e(3.0) == p(0.0, -1.0));
        assert!(e(3) == p(0, -1));
        assert!(e(4.0) == p(1.0, 0.0));
        assert!(e(4) == p(1, 0));
    }
    #[test]
    fn test_projection() {
        assert!(project(p(1.0, 0.0), p(5.0, 0.0)) == p(1.0, 0.0));
        assert!(project(p(1.0, 0.0), p(0.0, 5.0)) == p(0.0, 0.0));
        assert!(project(p(1.0, 1.0), p(1.0, 0.0)) == p(1.0, 0.0));
        assert!(project(p(6.0, 6.0), p(0.0, 1.0)) == p(0.0, 6.0));
        assert!(project(p(2.0, 6.0), p(0.0, 1.0)) == p(0.0, 6.0));
        assert!(project(p(-6.0, 6.0), p(0.0, 1.0)) == p(0.0, 6.0));
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
}
