mod glyph;
mod utility;

extern crate geo;
extern crate line_drawing;
extern crate num;
extern crate std;
extern crate termion;

// use assert2::{assert, check};
use crate::num::traits::Pow;
use geo::algorithm::euclidean_distance::EuclideanDistance;
use geo::algorithm::line_intersection::{line_intersection, LineIntersection};
use geo::{point, CoordNum, Point};
use std::char;
use std::cmp::min;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::io::{stdin, stdout, Write};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant};
use termion::color;
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

use glyph::*;
use utility::*;

// const player_jump_height: i32 = 3;
// const player_jump_hang_frames: i32 = 4;
const MAX_FPS: i32 = 60; // frames per second
const IDEAL_FRAME_DURATION_MS: u128 = (1000.0 / MAX_FPS as f32) as u128;
const PLAYER_COLOR: ColorName = ColorName::Red;
const PLAYER_HIGH_SPEED_COLOR: ColorName = ColorName::Blue;
const NUM_SAVED_PLAYER_POSES: i32 = 10;
const NUM_POSITIONS_TO_CHECK_PER_BLOCK_FOR_COLLISIONS: f32 = 8.0;

// a block every two ticks
const PLAYER_DEFAULT_MAX_SPEED_BPF: f32 = 0.5;
const DEFAULT_PLAYER_JUMP_DELTA_V: f32 = 1.0;
const DEFAULT_PLAYER_DASH_V: f32 = 5.0;
const DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY: f32 = 0.1;
const DEFAULT_PLAYER_ACCELERATION_FROM_TRACTION: f32 = 1.0;
const DEFAULT_PLAYER_COYOTE_TIME_DURATION_S: f32 = 0.1;
const DEFAULT_PLAYER_MAX_COYOTE_FRAMES: i32 =
    ((DEFAULT_PLAYER_COYOTE_TIME_DURATION_S * MAX_FPS as f32) + 1.0) as i32;
const PLAYER_SNAP_TO_GRID_ON_STOP: bool = false;
const PLAYER_SLOW_DOWN_MIDAIR: bool = false;

// These have no positional information
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Block {
    Air,
    Wall,
    Brick,
}

impl Block {
    fn glyph(&self) -> char {
        match self {
            Block::Air => ' ',
            Block::Wall => '█',
            Block::Brick => '▪',
        }
    }
    fn color(&self) -> ColorName {
        match self {
            Block::Air => ColorName::Black,
            Block::Wall => ColorName::White,
            Block::Brick => ColorName::White,
        }
    }

    fn subject_to_block_gravity(&self) -> bool {
        match self {
            Block::Air | Block::Wall => false,
            _ => true,
        }
    }
    fn wall_grabbable(&self) -> bool {
        match self {
            Block::Air => false,
            _ => true,
        }
    }
}

struct Game {
    grid: Vec<Vec<Block>>,             // (x,y), left to right, top to bottom
    output_buffer: Vec<Vec<Glyph>>,    // (x,y), left to right, top to bottom
    output_on_screen: Vec<Vec<Glyph>>, // (x,y), left to right, top to bottom
    terminal_size: (u16, u16),         // (width, height)
    prev_mouse_pos: (i32, i32),        // where mouse was last frame (if pressed)
    // last_pressed_key: Option<termion::event::Key>,
    running: bool,         // set false to quit
    selected_block: Block, // What the mouse places
    player_alive: bool,
    player_pos: Point<f32>,
    recent_player_poses: VecDeque<Point<f32>>,
    player_max_run_speed_bpf: f32,
    player_color_change_speed_threshold: f32,
    player_vel_bpf: Point<f32>,
    player_desired_direction: Point<i32>,
    player_jump_delta_v: f32,
    player_acceleration_from_gravity: f32,
    player_acceleration_from_traction: f32,
    player_remaining_coyote_frames: i32,
    player_max_coyote_frames: i32,
    player_color: ColorName,
    num_positions_per_block_to_check_for_collisions: f32,
}

impl Game {
    fn new(width: u16, height: u16) -> Game {
        Game {
            grid: vec![vec![Block::Air; height as usize]; width as usize],
            output_buffer: vec![vec![Glyph::from_char(' '); height as usize]; width as usize],
            output_on_screen: vec![vec![Glyph::from_char('x'); height as usize]; width as usize],
            terminal_size: (width, height),
            prev_mouse_pos: (1, 1),
            running: true,
            selected_block: Block::Wall,
            player_alive: false,
            player_pos: p(0.0, 0.0),
            recent_player_poses: VecDeque::<Point<f32>>::new(),
            player_max_run_speed_bpf: PLAYER_DEFAULT_MAX_SPEED_BPF,
            player_color_change_speed_threshold: magnitude(p(
                PLAYER_DEFAULT_MAX_SPEED_BPF,
                DEFAULT_PLAYER_JUMP_DELTA_V,
            )),
            player_vel_bpf: Point::<f32>::new(0.0, 0.0),
            player_desired_direction: p(0, 0),
            player_jump_delta_v: DEFAULT_PLAYER_JUMP_DELTA_V,
            player_acceleration_from_gravity: DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY,
            player_acceleration_from_traction: DEFAULT_PLAYER_ACCELERATION_FROM_TRACTION,
            player_remaining_coyote_frames: DEFAULT_PLAYER_MAX_COYOTE_FRAMES,
            player_max_coyote_frames: DEFAULT_PLAYER_MAX_COYOTE_FRAMES,
            player_color: PLAYER_COLOR,
            num_positions_per_block_to_check_for_collisions:
                NUM_POSITIONS_TO_CHECK_PER_BLOCK_FOR_COLLISIONS,
        }
    }

    fn set_player_jump_delta_v(&mut self, delta_v: f32) {
        self.player_jump_delta_v = delta_v;
        self.update_player_color_change_speed_thresh();
    }

    fn set_player_max_run_speed(&mut self, speed: f32) {
        self.player_max_run_speed_bpf = speed;
        self.update_player_color_change_speed_thresh();
    }

    fn update_player_color_change_speed_thresh(&mut self) {
        self.player_color_change_speed_threshold =
            magnitude(p(self.player_max_run_speed_bpf, self.player_jump_delta_v));
    }

    fn screen_to_world(&self, terminal_position: &(u16, u16)) -> (i32, i32) {
        // terminal indexes from 1, and the y axis goes top to bottom
        (
            terminal_position.0 as i32 - 1,
            self.terminal_size.1 as i32 - terminal_position.1 as i32,
        )
    }

    fn world_to_screen(&self, world_position: &(i32, i32)) -> (u16, u16) {
        // terminal indexes from 1, and the y axis goes top to bottom
        // world indexes from 0, origin at bottom left
        (
            world_position.0 as u16 + 1,
            (self.terminal_size.1 as i32 - world_position.1) as u16,
        )
    }
    fn get_block(&self, pos: Point<i32>) -> Block {
        return self.grid[pos.x() as usize][pos.y() as usize];
    }
    fn set_block(&mut self, pos: Point<i32>, block: Block) {
        self.grid[pos.x() as usize][pos.y() as usize] = block;
    }

    fn place_line_of_blocks(&mut self, pos0: (i32, i32), pos1: (i32, i32), block: Block) {
        for (x1, y1) in line_drawing::Bresenham::new(pos0, pos1) {
            self.grid[x1 as usize][y1 as usize] = block;
        }
    }

    fn place_block(&mut self, pos: Point<i32>, block: Block) {
        self.grid[pos.x() as usize][pos.y() as usize] = block;
    }

    fn draw_visual_braille_line(
        &mut self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        color: ColorName,
    ) {
        let squares_to_place =
            Glyph::get_glyphs_for_colored_braille_line(start_pos, end_pos, color);

        let start_grid_square = snap_to_grid(start_pos);
        let end_grid_square = snap_to_grid(end_pos);
        let grid_diagonal = end_grid_square - start_grid_square;
        let grid_width = grid_diagonal.x().abs() + 1;
        let grid_height = grid_diagonal.y().abs() + 1;
        let bottom_square_y = min(start_grid_square.y(), end_grid_square.y());
        let left_square_x = min(start_grid_square.x(), end_grid_square.x());

        for i in 0..squares_to_place.len() {
            for j in 0..squares_to_place[0].len() {
                if let Some(new_glyph) = squares_to_place[i][j] {
                    let grid_square = p(left_square_x + i as i32, bottom_square_y + j as i32);
                    if !self.in_world(grid_square) {
                        continue;
                    }
                    let grid_glyph =
                        &mut self.output_buffer[grid_square.x() as usize][grid_square.y() as usize];
                    if Glyph::is_braille(grid_glyph.character) {
                        let combined_braille =
                            Glyph::add_braille(grid_glyph.character, new_glyph.character);
                        *grid_glyph = new_glyph;
                        grid_glyph.character = combined_braille;
                    } else {
                        *grid_glyph = new_glyph;
                    }
                }
            }
        }
    }

    fn clear(&mut self) {
        let (width, height) = termion::terminal_size().unwrap();
        self.grid = vec![vec![Block::Air; height as usize]; width as usize];
    }

    fn place_player(&mut self, x: f32, y: f32) {
        if self.player_alive {
            self.kill_player();
        }
        let grid_point = snap_to_grid(p(x, y));
        self.player_vel_bpf = p(0.0, 0.0);
        self.player_desired_direction = p(0, 0);
        self.player_pos = p(x, y);
        self.player_alive = true;
        self.player_remaining_coyote_frames = 0;
    }
    fn player_jump(&mut self) {
        if self.player_is_grabbing_wall() || self.player_is_running_up_wall() {
            let wall_direction = sign(self.player_desired_direction);
            self.player_vel_bpf
                .add_assign(floatify(-wall_direction) * self.player_jump_delta_v);
            self.player_desired_direction = -wall_direction;
        }
        self.player_vel_bpf.set_y(self.player_jump_delta_v);
    }
    fn player_jump_if_possible(&mut self) {
        if self.player_is_supported()
            || self.player_is_grabbing_wall()
            || self.player_is_running_up_wall()
        {
            self.player_jump();
        }
    }

    fn player_dash(&mut self) {
        if self.player_desired_direction != p(0, 0) {
            self.player_vel_bpf = floatify(self.player_desired_direction) * DEFAULT_PLAYER_DASH_V;
        }
    }

    fn handle_input(&mut self, evt: termion::event::Event) {
        match evt {
            Event::Key(ke) => match ke {
                Key::Char('q') => self.running = false,
                Key::Char('1') => self.selected_block = Block::Air,
                Key::Char('2') => self.selected_block = Block::Wall,
                Key::Char('3') => self.selected_block = Block::Brick,
                Key::Char('c') => self.clear(),
                Key::Char('r') => self.place_player(
                    self.terminal_size.0 as f32 / 2.0,
                    self.terminal_size.1 as f32 / 2.0,
                ),
                Key::Char(' ') => self.player_jump_if_possible(),
                Key::Char('f') => self.player_dash(),
                Key::Char('w') | Key::Up => self.player_desired_direction = p(0, 1),
                Key::Char('a') | Key::Left => self.player_desired_direction = p(-1, 0),
                Key::Char('s') | Key::Down => self.player_desired_direction = p(0, -1),
                Key::Char('d') | Key::Right => self.player_desired_direction = p(1, 0),
                _ => {}
            },
            Event::Mouse(me) => match me {
                MouseEvent::Press(MouseButton::Left, term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.place_block(p(x, y), self.selected_block);
                    self.prev_mouse_pos = (x, y);
                }
                MouseEvent::Press(MouseButton::Right, term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.place_player(x as f32, y as f32);
                }
                MouseEvent::Hold(term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.place_line_of_blocks(self.prev_mouse_pos, (x, y), self.selected_block);
                    self.prev_mouse_pos = (x, y);
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn tick_physics(&mut self) {
        self.apply_gravity_to_blocks();
        if self.player_alive {
            self.apply_player_motion();
        }
    }

    fn update_output_buffer(&mut self) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        for x in 0..width {
            for y in 0..height {
                self.output_buffer[x][y] = Glyph::from_char(self.grid[x][y].glyph());
            }
        }

        if magnitude(self.player_vel_bpf) > self.player_color_change_speed_threshold {
            if let Some(&last_pos) = self.recent_player_poses.get(0) {
                // draw all corners
                let offsets = [p(0.5, 0.5), p(-0.5, 0.5), p(-0.5, -0.5), p(0.5, -0.5)];
                for offset in offsets {
                    self.draw_visual_braille_line(
                        last_pos + offset,
                        self.player_pos + offset,
                        ColorName::Blue,
                    );
                }
            }
        }

        let player_glyphs = self.get_player_glyphs();
        let grid_pos = snap_to_grid(self.player_pos);

        for i in 0..player_glyphs.len() {
            for j in 0..player_glyphs[i].len() {
                if let Some(glyph) = player_glyphs[i][j] {
                    let x = grid_pos.x() - 1 + i as i32;
                    let y = grid_pos.y() - 1 + j as i32;

                    if self.in_world(p(x, y)) {
                        self.output_buffer[x as usize][y as usize] = glyph;
                    }
                }
            }
        }
    }
    fn get_player_glyphs(&self) -> Vec<Vec<Option<Glyph>>> {
        if magnitude(self.player_vel_bpf) > self.player_color_change_speed_threshold {
            Glyph::get_glyphs_for_colored_floating_square(self.player_pos, PLAYER_HIGH_SPEED_COLOR)
        } else {
            Glyph::get_glyphs_for_colored_floating_square(self.player_pos, self.player_color)
        }
    }

    fn get_buffered_glyph(&self, pos: Point<i32>) -> &Glyph {
        return &self.output_buffer[pos.x() as usize][pos.y() as usize];
    }
    fn get_glyph_on_screen(&self, pos: Point<i32>) -> &Glyph {
        return &self.output_on_screen[pos.x() as usize][pos.y() as usize];
    }

    fn update_screen(
        &mut self,
        stdout: &mut MouseTerminal<termion::raw::RawTerminal<std::io::Stdout>>,
    ) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        // Now update the graphics where applicable
        for x in 0..width {
            for y in 0..height {
                if self.output_buffer[x][y] != self.output_on_screen[x][y] {
                    let (term_x, term_y) = self.world_to_screen(&(x as i32, y as i32));
                    write!(stdout, "{}", termion::cursor::Goto(term_x, term_y)).unwrap();
                    write!(stdout, "{}", self.output_buffer[x][y].to_string()).unwrap();
                }
            }
        }
        write!(stdout, "{}", termion::cursor::Goto(1, 1),).unwrap();
        stdout.flush().unwrap();
        self.output_on_screen = self.output_buffer.clone();
    }

    fn apply_gravity_to_blocks(&mut self) {
        // We want to count from bottom to top, because things fall down
        for x in 0..self.terminal_size.0 as usize {
            for y in 0..self.terminal_size.1 as usize {
                let block = self.grid[x][y];
                if block.subject_to_block_gravity() {
                    let is_bottom_row = y == 0;
                    let has_direct_support = !is_bottom_row && self.grid[x][y - 1] != Block::Air;
                    if is_bottom_row {
                        self.grid[x][y] = Block::Air;
                    } else if !has_direct_support {
                        self.grid[x][y - 1] = block;
                        self.grid[x][y] = Block::Air;
                    }
                }
            }
        }
    }

    fn kill_player(&mut self) {
        self.set_block(snap_to_grid(self.player_pos), Block::Air);
        self.player_alive = false;
    }

    fn player_is_grabbing_wall(&self) -> bool {
        self.player_is_pressing_against_wall()
            && self.player_vel_bpf.y() <= 0.0
            && !self.player_is_standing_on_block()
    }

    fn player_is_running_up_wall(&self) -> bool {
        self.player_is_pressing_against_wall() && self.player_vel_bpf.y() > 0.0
    }

    fn player_is_pressing_against_wall(&self) -> bool {
        if let Some(block) = self.get_block_relative_to_player(self.player_desired_direction) {
            return block.wall_grabbable();
        }
        return false;
    }

    fn apply_player_acceleration_from_wall_friction(&mut self) {
        let direction_of_acceleration = -self.player_vel_bpf.y().sign();
        let delta_v = direction_of_acceleration * self.player_acceleration_from_traction;
        if delta_v.abs() > self.player_vel_bpf.y().abs() {
            self.player_vel_bpf.set_y(0.0);
        } else {
            self.player_vel_bpf.add_assign(p(0.0, delta_v));
        }
    }

    fn apply_player_acceleration_from_floor_traction(&mut self) {
        let start_x_vel = self.player_vel_bpf.x();

        let want_stop = self.player_desired_direction.x() == 0 && start_x_vel != 0.0;

        let want_go_fast = (start_x_vel.abs() > 0.0
            && start_x_vel.sign() == self.player_desired_direction.x().sign() as f32)
            || (start_x_vel == 0.0 && self.player_desired_direction.x() != 0);

        let want_turn_around =
            start_x_vel.abs() > 0.0 && self.player_desired_direction.x() as f32 * start_x_vel < 0.0;

        let desired_acceleration_direction;
        if want_stop {
            desired_acceleration_direction = -start_x_vel.sign() as i32;
        } else {
            desired_acceleration_direction = self.player_desired_direction.x().sign();
        }

        let started_above_max_speed = start_x_vel.abs() > self.player_max_run_speed_bpf;

        let midair = !self.player_is_supported();

        let mut end_x_vel = start_x_vel;
        if started_above_max_speed {
            if !(midair && want_go_fast) {
                // slowing down to the max speed automatically
                let delta_vx = -start_x_vel.sign() * self.player_acceleration_from_traction;
                let is_slowdown_overshoot =
                    (start_x_vel + delta_vx).abs() < self.player_max_run_speed_bpf;
                if want_go_fast && is_slowdown_overshoot {
                    end_x_vel = start_x_vel.sign() as f32 * self.player_max_run_speed_bpf;
                } else {
                    end_x_vel = start_x_vel + delta_vx;
                }
            }
        } else {
            // can speed up or slow down. maybe limited by max speed
            let delta_vx =
                desired_acceleration_direction as f32 * self.player_acceleration_from_traction;
            let is_speedup_overshoot =
                (start_x_vel + delta_vx).abs() > self.player_max_run_speed_bpf;
            if is_speedup_overshoot && !midair {
                end_x_vel = desired_acceleration_direction as f32 * self.player_max_run_speed_bpf;
            } else {
                end_x_vel = start_x_vel + delta_vx;
            }
        }

        let changed_direction = start_x_vel * end_x_vel < 0.0;
        let is_stop_overshoot = want_stop && changed_direction;

        if is_stop_overshoot {
            end_x_vel = 0.0;
        }

        if end_x_vel == 0.0 && PLAYER_SNAP_TO_GRID_ON_STOP {
            self.player_pos
                .set_x(snap_to_grid(self.player_pos).x() as f32);
        }

        self.player_vel_bpf.set_x(end_x_vel);
    }

    fn apply_player_acceleration_from_gravity(&mut self) {
        self.player_vel_bpf
            .add_assign(p(0.0, -self.player_acceleration_from_gravity));
    }

    fn apply_player_motion(&mut self) {
        self.update_player_acceleration();

        let start_point = self.player_pos;
        let mut remaining_step: Point<f32> = self.player_vel_bpf;
        let mut current_start_point = start_point;
        let actual_endpoint: Point<f32>;
        let mut current_target = start_point + remaining_step;
        loop {
            if let Some(collision) = self.movecast(current_start_point, current_target) {
                remaining_step.add_assign(-(collision.pos - current_start_point));

                if collision.normal.x() != 0 {
                    self.player_vel_bpf.set_x(0.0);
                    remaining_step = project(remaining_step, p(0.0, 1.0));
                }
                if collision.normal.y() != 0 {
                    self.player_vel_bpf.set_y(0.0);
                    remaining_step = project(remaining_step, p(1.0, 0.0));
                }
                current_start_point = collision.pos;
                current_target = current_start_point + remaining_step;
            } else {
                // should exit loop after this else
                actual_endpoint = current_target;
                break;
            }
        }

        let step_taken: Point<f32>;

        if !self.in_world(snap_to_grid(actual_endpoint)) {
            // Player went out of bounds and died
            self.kill_player();
            return;
        } else {
            // no collision, and in world
            step_taken = actual_endpoint - self.player_pos;
            self.save_recent_player_pose(self.player_pos);
            self.player_pos = actual_endpoint;
        }

        // moved vertically => instant empty charge
        if step_taken.y() != 0.0 {
            self.player_remaining_coyote_frames = 0;
        }
        if self.player_is_standing_on_block() {
            self.player_remaining_coyote_frames = self.player_max_coyote_frames;
        } else if self.player_remaining_coyote_frames > 0 {
            self.player_remaining_coyote_frames -= 1;
        }
    }

    fn save_recent_player_pose(&mut self, pos: Point<f32>) {
        self.recent_player_poses.push_front(pos);
        while self.recent_player_poses.len() > NUM_SAVED_PLAYER_POSES as usize {
            self.recent_player_poses.pop_back();
        }
    }

    fn update_player_acceleration(&mut self) {
        if self.player_is_grabbing_wall() && self.player_vel_bpf.y() <= 0.0 {
            self.apply_player_acceleration_from_wall_friction();
        } else {
            if !self.player_is_supported() {
                self.apply_player_acceleration_from_gravity();
            }
            self.apply_player_acceleration_from_floor_traction();
        }
    }

    // Where the player can move to in a line
    // tries to draw a line in air
    // returns None if out of bounds
    // returns the start position if start is not Block::Air
    fn movecast(&self, start_pos: Point<f32>, end_pos: Point<f32>) -> Option<MovecastCollision> {
        let ideal_step = end_pos - start_pos;

        let collision_checks_per_block_travelled =
            self.num_positions_per_block_to_check_for_collisions;
        let num_points_to_check =
            (magnitude(ideal_step) * collision_checks_per_block_travelled).floor() as i32;
        let mut intermediate_player_positions_to_check = Vec::<Point<f32>>::new();
        for i in 0..num_points_to_check {
            intermediate_player_positions_to_check.push(
                start_pos
                    + direction(ideal_step) * (i as f32 / collision_checks_per_block_travelled),
            );
        }
        // needed for very small steps
        intermediate_player_positions_to_check.push(end_pos);
        for point_to_check in &intermediate_player_positions_to_check {
            let overlapping_squares =
                grid_squares_overlapped_by_floating_unit_square(*point_to_check);
            let mut collisions = Vec::<MovecastCollision>::new();
            for overlapping_square in &overlapping_squares {
                if self.in_world(*overlapping_square)
                    && self.get_block(*overlapping_square) == Block::Wall
                {
                    if let Some(collision) =
                        single_block_movecast(start_pos, *point_to_check, *overlapping_square)
                    {
                        collisions.push(collision);
                    } else {
                        panic!("Started inside a block");
                    }
                }
            }
            if !collisions.is_empty() {
                collisions.sort_by(|a, b| {
                    let a_dist = a.pos.euclidean_distance(&start_pos);
                    let b_dist = b.pos.euclidean_distance(&start_pos);
                    a_dist.partial_cmp(&b_dist).unwrap()
                });
                let closest_collision_to_start = collisions[0];
                return Some(closest_collision_to_start);
            }
        }
        return None;
    }

    fn player_is_supported(&self) -> bool {
        return self.player_is_standing_on_block() || self.player_remaining_coyote_frames > 0;
    }

    fn player_is_standing_on_block(&self) -> bool {
        match self.get_block_below_player() {
            None | Some(Block::Air) => false,
            _ => true,
        }
    }

    fn get_block_below_player(&self) -> Option<Block> {
        return self.get_block_relative_to_player(p(0, -1));
    }

    fn get_block_relative_to_player(&self, rel_pos: Point<i32>) -> Option<Block> {
        let target_pos = snap_to_grid(self.player_pos) + rel_pos;
        if self.player_alive && self.in_world(target_pos) {
            return Some(self.get_block(target_pos));
        }
        return None;
    }

    fn in_world(&self, pos: Point<i32>) -> bool {
        return pos.x() >= 0
            && pos.x() < self.terminal_size.0 as i32
            && pos.y() >= 0
            && pos.y() < self.terminal_size.1 as i32;
    }
    fn init_world(&mut self) {
        self.set_player_jump_delta_v(1.0);
        self.player_acceleration_from_gravity = 0.05;
        self.player_acceleration_from_traction = 0.6;
        self.set_player_max_run_speed(0.7);

        let bottom_left = (
            (self.terminal_size.0 / 5) as i32,
            (self.terminal_size.1 / 4) as i32,
        );
        self.place_line_of_blocks(
            bottom_left,
            ((4 * self.terminal_size.0 / 5) as i32, bottom_left.1),
            Block::Wall,
        );
        self.place_line_of_blocks(
            bottom_left,
            (bottom_left.0, 3 * (self.terminal_size.1 / 4) as i32),
            Block::Wall,
        );
        self.place_player(
            self.terminal_size.0 as f32 / 2.0,
            self.terminal_size.1 as f32 / 2.0,
        );
    }
}

fn main() {
    let stdin = stdin();
    let (width, height) = termion::terminal_size().unwrap();
    let mut game = Game::new(width, height);
    let mut stdout = MouseTerminal::from(stdout().into_raw_mode().unwrap());

    write!(
        stdout,
        "{}{}q to exit.  c to clear.  Mouse to draw.  Begin!",
        termion::clear::All,
        termion::cursor::Goto(1, 1)
    )
    .unwrap();
    stdout.flush().unwrap();

    let (tx, rx) = channel();

    // Separate thread for reading input
    thread::spawn(move || {
        for c in stdin.events() {
            let evt = c.unwrap();
            tx.send(evt).unwrap();
        }
    });

    // time saver
    game.init_world();

    while game.running {
        let start_time = Instant::now();
        while let Ok(evt) = rx.try_recv() {
            game.handle_input(evt);
        }
        game.tick_physics();
        game.update_output_buffer();
        game.update_screen(&mut stdout);
        let tick_duration_so_far_ms = start_time.elapsed().as_millis();
        if tick_duration_so_far_ms < IDEAL_FRAME_DURATION_MS {
            thread::sleep(Duration::from_millis(
                (IDEAL_FRAME_DURATION_MS - tick_duration_so_far_ms) as u64,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;

    fn set_up_game() -> Game {
        Game::new(30, 30)
    }

    fn set_up_just_player() -> Game {
        let mut game = set_up_game();
        game.place_player(15.0, 11.0);
        return game;
    }

    fn set_up_player_on_platform() -> Game {
        let mut game = set_up_just_player();
        game.place_line_of_blocks((10, 10), (20, 10), Block::Wall);
        return game;
    }

    fn set_up_player_on_platform_in_box() -> Game {
        let mut game = set_up_player_on_platform();
        let xmax = game.grid.len() as i32 - 1;
        let ymax = game.grid[0].len() as i32 - 1;
        game.place_line_of_blocks((0, 0), (xmax, 0), Block::Wall);
        game.place_line_of_blocks((xmax, 0), (xmax, ymax), Block::Wall);
        game.place_line_of_blocks((xmax, ymax), (0, ymax), Block::Wall);
        game.place_line_of_blocks((0, ymax), (0, 0), Block::Wall);
        return game;
    }

    fn set_up_player_hanging_on_wall_on_left() -> Game {
        let mut game = set_up_just_player();
        game.place_line_of_blocks((14, 0), (14, 20), Block::Wall);
        game.player_desired_direction.set_x(-1);
        return game;
    }

    #[test]
    fn test_placement_and_gravity() {
        let mut game = Game::new(30, 30);
        game.place_block(p(10, 10), Block::Wall);
        game.place_block(p(20, 10), Block::Brick);

        assert!(game.grid[10][10] == Block::Wall);
        assert!(game.grid[20][10] == Block::Brick);
        game.tick_physics();

        assert!(game.grid[10][10] == Block::Wall);
        assert!(game.grid[20][10] == Block::Air);

        assert!(game.grid[10][9] == Block::Air);
        assert!(game.grid[20][9] == Block::Brick);
    }

    #[test]
    fn test_place_player() {
        let mut game = Game::new(30, 30);
        // TODO: should these be variables?  Or should I just hardcode them?
        let (x1, y1, x2, y2) = (15.0, 11.0, 12.0, 5.0);
        assert!(game.player_alive == false);
        game.place_player(x1, y1);

        assert!(game.player_pos == p(x1, y1));
        assert!(game.player_alive == true);

        game.place_player(x2, y2);
        assert!(game.player_pos == p(x2, y2));
        assert!(game.player_alive == true);
    }

    #[test]
    fn test_player_dies_when_falling_off_screen() {
        let mut game = Game::new(30, 30);
        game.place_player(15.0, 0.0);
        game.player_acceleration_from_gravity = 1.0;
        game.player_remaining_coyote_frames = 0;
        game.tick_physics();
        assert!(game.player_alive == false);
    }

    #[test]
    fn test_single_block_movecast_no_move() {
        let point = p(0.0, 0.0);
        let p_wall = p(5, 5);

        assert!(single_block_movecast(point, point, p_wall) == None);
    }

    #[test]
    fn test_movecast_no_move() {
        let game = Game::new(30, 30);
        let point = p(0.0, 0.0);

        assert!(game.movecast(point, point) == None);
    }

    #[test]
    fn test_movecast_horizontal_hit() {
        let mut game = Game::new(30, 30);
        let p_wall = p(5, 0);
        game.place_block(p_wall, Block::Wall);

        let p1 = floatify(p_wall) + p(-2.0, 0.0);
        let p2 = floatify(p_wall) + p(2.0, 0.0);
        let result = game.movecast(p1, p2);

        assert!(result != None);
        assert!(result.unwrap().pos == floatify(p_wall) + p(-1.0, 0.0));
        assert!(result.unwrap().normal == p(-1, 0));
    }

    #[test]
    fn test_movecast_vertical_hit() {
        let mut game = Game::new(30, 30);
        let p_wall = p(15, 10);
        game.place_block(p_wall, Block::Wall);

        let p1 = floatify(p_wall) + p(0.0, -1.1);
        let p2 = floatify(p_wall);
        let result = game.movecast(p1, p2);

        assert!(result != None);
        assert!(result.unwrap().pos == floatify(p_wall) + p(0.0, -1.0));
        assert!(result.unwrap().normal == p(0, -1));
    }

    #[test]
    fn test_movecast_end_slightly_overlapping_a_block() {
        let mut game = Game::new(30, 30);
        let p_wall = p(15, 10);
        game.place_block(p_wall, Block::Wall);

        let p1 = floatify(p_wall) + p(0.0, 1.5);
        let p2 = floatify(p_wall) + p(0.0, 0.999999);
        let result = game.movecast(p1, p2);

        assert!(result != None);
        assert!(result.unwrap().pos == floatify(p_wall) + p(0.0, 1.0));
        assert!(result.unwrap().normal == p(0, 1));
    }

    #[test]
    fn test_movecast() {
        let mut game = Game::new(30, 30);
        game.place_line_of_blocks((10, 10), (20, 10), Block::Wall);

        assert!(
            game.movecast(p(15.0, 9.0), p(17.0, 11.0))
                == Some(MovecastCollision {
                    pos: p(15.0, 9.0),
                    normal: p(0, -1)
                })
        );
        assert!(
            game.movecast(p(15.0, 9.0), p(17.0, 110.0))
                == Some(MovecastCollision {
                    pos: p(15.0, 9.0),
                    normal: p(0, -1)
                })
        );
        assert!(game.movecast(p(1.0, 9.0), p(-17.0, 9.0)) == None);
        assert!(game.movecast(p(15.0, 9.0), p(17.0, -11.0)) == None);
    }

    #[test]
    fn test_movecast_ignore_player() {
        let game = set_up_player_on_platform();

        assert!(game.movecast(p(15.0, 11.0), p(15.0, 13.0)) == None);
    }

    #[test]
    fn test_in_world_check() {
        let game = Game::new(30, 30);
        assert!(game.in_world(p(0, 0)));
        assert!(game.in_world(p(29, 29)));
        assert!(!game.in_world(p(30, 30)));
        assert!(!game.in_world(p(10, -1)));
        assert!(!game.in_world(p(-1, 10)));
    }

    #[test]
    fn test_move_player() {
        let mut game = set_up_player_on_platform();
        game.player_max_run_speed_bpf = 1.0;
        game.player_desired_direction.set_x(1);

        game.tick_physics();

        assert!(game.player_pos == p(16.0, 11.0));

        game.place_player(15.0, 11.0);
        assert!(game.player_desired_direction.x() == 0);
        game.player_desired_direction.set_x(-1);

        game.tick_physics();
        game.tick_physics();

        assert!(game.player_pos == p(13.0, 11.0));
    }

    #[test]
    fn test_stop_on_collision() {
        let mut game = set_up_player_on_platform();
        game.place_block(round(game.player_pos) + p(1, 0), Block::Wall);
        game.player_max_run_speed_bpf = 1.0;
        game.player_desired_direction.set_x(1);

        game.tick_physics();

        assert!(game.player_pos == p(15.0, 11.0));
        assert!(game.player_vel_bpf.x() == 0.0);
    }

    #[test]
    fn test_snap_to_object_on_collision() {
        let mut game = set_up_player_on_platform();
        game.place_block(round(game.player_pos) + p(2, 0), Block::Wall);
        game.player_pos.add_assign(p(0.999, 0.0));
        game.player_desired_direction.set_x(1);
        game.player_vel_bpf.set_x(5.0);

        game.tick_physics();

        assert!(game.player_pos == p(16.0, 11.0));
        assert!(game.grid[17][11] == Block::Wall);
    }

    #[test]
    fn test_move_player_slowly() {
        let mut game = set_up_player_on_platform();
        game.player_max_run_speed_bpf = 0.49;
        game.player_acceleration_from_traction = 999.9; // rilly fast
        game.player_desired_direction.set_x(1);

        game.tick_physics();

        assert!(game.player_pos.x() > 15.0);
        assert!(game.player_pos.x() < 15.5);

        game.tick_physics();

        assert!(game.player_pos.x() > 15.5);
        assert!(game.player_pos.x() < 16.0);
    }

    #[test]
    fn test_move_player_quickly() {
        let mut game = set_up_player_on_platform();
        game.player_max_run_speed_bpf = 2.0;
        game.player_desired_direction.set_x(1);

        game.tick_physics();
        game.tick_physics();
        assert!(game.player_pos.x() > 17.0);
        game.tick_physics();
        assert!(game.player_pos.x() > 19.0);
    }

    #[test]
    fn test_fast_player_collision_between_frames() {
        let mut game = set_up_player_on_platform();
        // Player should not teleport through this block
        game.place_block(p(16, 11), Block::Wall);
        game.player_max_run_speed_bpf = 2.0;
        game.player_desired_direction.set_x(1);

        game.tick_physics();
        assert!(game.player_pos == p(15.0, 11.0));
        assert!(game.player_vel_bpf.x() == 0.0);
    }

    #[test]
    fn test_can_jump() {
        let mut game = set_up_player_on_platform();
        let start_pos = game.player_pos;

        game.player_jump();
        game.tick_physics();
        assert!(game.player_pos.y() > start_pos.y());
    }

    #[test]
    fn test_player_gravity() {
        let mut game = Game::new(30, 30);
        game.place_player(15.0, 11.0);
        game.player_acceleration_from_gravity = 1.0;
        game.player_remaining_coyote_frames = 0;

        game.tick_physics();

        assert!(game.player_pos.y() < 11.0);
    }

    #[test]
    fn test_land_after_jump() {
        let mut game = set_up_player_on_platform();
        let start_pos = game.player_pos;

        game.player_jump();
        for _ in 0..50 {
            game.tick_physics();
        }
        assert!(game.player_pos.y() == start_pos.y());
    }

    #[test]
    fn test_slide_on_angled_collision() {
        let mut game = Game::new(30, 30);
        game.place_line_of_blocks((10, 10), (20, 10), Block::Wall);
        game.place_line_of_blocks((14, 10), (14, 20), Block::Wall);

        game.place_player(15.0, 11.0);
        game.player_vel_bpf = p(-2.0, 2.0);
        game.tick_physics();
        assert!(game.player_vel_bpf.x() == 0.0);
        assert!(game.player_vel_bpf.y() > 0.0);
    }

    #[test]
    fn test_decellerate_when_already_moving_faster_than_max_speed() {
        let mut game = set_up_player_on_platform();
        game.player_max_run_speed_bpf = 1.0;
        game.player_vel_bpf.set_x(5.0);
        game.player_desired_direction.set_x(1);
        game.tick_physics();
        assert!(game.player_vel_bpf.x() > game.player_max_run_speed_bpf);
        assert!(game.player_vel_bpf.x() < 5.0);
    }

    #[test]
    fn test_no_double_jump() {
        let mut game = set_up_player_on_platform();
        game.player_jump_if_possible();
        game.tick_physics();
        let vel_y_before_second_jump = game.player_vel_bpf.y();
        game.player_jump_if_possible();
        game.tick_physics();
        assert!(game.player_vel_bpf.y() < vel_y_before_second_jump);
    }

    #[test]
    fn test_respawn_button() {
        let mut game = Game::new(30, 30);
        game.handle_input(Event::Key(Key::Char('r')));
        assert!(game.player_alive);
        assert!(game.player_pos == p(15.0, 15.0));
    }

    #[test]
    fn test_wall_grab() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.tick_physics();
        assert!(game.player_vel_bpf.y() == 0.0);
    }

    #[test]
    fn test_simple_wall_jump() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_jump_if_possible();
        game.tick_physics();
        assert!(game.player_vel_bpf.y() > 0.0);
        assert!(game.player_vel_bpf.x() > 0.0);
    }

    #[test]
    fn test_no_running_up_walls_immediately_after_spawn() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_vel_bpf = p(-1.0, 1.0);
        let start_vel_y = game.player_vel_bpf.y();
        game.tick_physics();
        assert!(game.player_vel_bpf.y() < start_vel_y);
    }

    #[test]
    fn test_dont_grab_wall_while_moving_up() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_vel_bpf.set_y(1.0);
        assert!(!game.player_is_grabbing_wall());
    }

    #[test]
    fn test_no_friction_when_sliding_up_wall() {
        let mut game = set_up_player_on_platform();
        let wall_x = game.player_pos.x() as i32 - 2;
        game.place_line_of_blocks((wall_x, 0), (wall_x, 20), Block::Wall);
        game.player_desired_direction.set_x(-1);
        game.player_pos.add_assign(p(0.0, 2.0));
        game.player_pos.set_x(wall_x as f32 + 1.0);
        game.player_vel_bpf = p(-3.0, 3.0);
        game.player_acceleration_from_gravity = 0.0;

        let start_y_vel = game.player_vel_bpf.y();
        let start_y_pos = game.player_pos.y();
        game.tick_physics();
        let end_y_vel = game.player_vel_bpf.y();
        let end_y_pos = game.player_pos.y();

        assert!(start_y_vel == end_y_vel);
        assert!(start_y_pos != end_y_pos);
    }

    #[test]
    fn test_coyote_frames() {
        let mut game = set_up_just_player();
        let start_pos = game.player_pos;
        game.player_acceleration_from_gravity = 1.0;
        game.player_max_coyote_frames = 1;
        game.player_remaining_coyote_frames = 1;
        game.tick_physics();
        assert!(game.player_pos.y() == start_pos.y());
        game.tick_physics();
        assert!(game.player_pos.y() < start_pos.y());
    }

    #[test]
    fn test_coyote_frames_dont_assist_jump() {
        let mut game1 = set_up_player_on_platform();
        let mut game2 = set_up_player_on_platform();

        game1.player_remaining_coyote_frames = 0;
        game1.player_jump_if_possible();
        game2.player_jump_if_possible();
        game1.tick_physics();
        game2.tick_physics();

        // Second tick after jump is where coyote physics may come into play
        game1.tick_physics();
        game2.tick_physics();

        assert!(game1.player_vel_bpf.y() == game2.player_vel_bpf.y());
    }

    #[test]
    fn test_player_does_not_spawn_with_coyote_frames() {
        let mut game = set_up_just_player();
        let start_pos = game.player_pos;
        game.player_acceleration_from_gravity = 1.0;
        game.tick_physics();
        assert!(game.player_pos.y() != start_pos.y());
    }

    #[test]
    fn test_wall_jump_while_running_up_wall() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_vel_bpf.set_y(1.0);
        game.player_jump_if_possible();
        assert!(game.player_vel_bpf.x() > 0.0);
    }

    #[test]
    fn test_draw_to_output_buffer() {
        let mut game = set_up_player_on_platform();
        game.update_output_buffer();
        assert!(
            game.get_buffered_glyph(snap_to_grid(game.player_pos))
                .character
                == EIGHTH_BLOCKS_FROM_LEFT[8]
        );
        assert!(
            game.get_buffered_glyph(snap_to_grid(game.player_pos + p(0.0, -1.0)))
                .character
                == Block::Wall.glyph()
        );
    }

    #[test]
    fn test_horizontal_sub_glyph_positioning_on_left() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(-0.2, 0.0));
        game.update_output_buffer();

        let left_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos + p(-1.0, 0.0)));
        let right_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[7]);
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[7]);
    }

    #[test]
    fn test_horizontal_sub_glyph_positioning_on_left_above_rounding_point() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(-0.49, 0.0));
        game.update_output_buffer();

        let left_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos + p(-1.0, 0.0)));
        let right_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[5]);
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[5]);
    }

    #[test]
    fn test_horizontal_sub_glyph_positioning_on_left_rounding_down() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(-0.5, 0.0));
        game.update_output_buffer();

        let left_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos + p(-1.0, 0.0)));
        let right_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[4]);
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[4]);
    }

    #[test]
    fn test_vertical_sub_glyph_positioning_upwards() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(0.0, 0.49));
        game.update_output_buffer();

        let top_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos + p(0.0, 1.0)));
        let bottom_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos));
        assert!(top_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[4]);
        assert!(top_glyph.fg_color == PLAYER_COLOR);
        assert!(top_glyph.bg_color == ColorName::Black);
        assert!(bottom_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[4]);
        assert!(bottom_glyph.fg_color == ColorName::Black);
        assert!(bottom_glyph.bg_color == PLAYER_COLOR);
    }

    #[test]
    fn test_vertical_sub_glyph_positioning_downwards() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(0.0, -0.2));
        game.update_output_buffer();

        let top_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos));
        let bottom_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos + p(0.0, -1.0)));
        assert!(top_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[7]);
        assert!(top_glyph.fg_color == PLAYER_COLOR);
        assert!(top_glyph.bg_color == ColorName::Black);
        assert!(bottom_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[7]);
        assert!(bottom_glyph.fg_color == ColorName::Black);
        assert!(bottom_glyph.bg_color == PLAYER_COLOR);
    }

    #[test]
    fn test_player_glyph_when_rounding_to_zero_for_both_axes() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(-0.24, 0.01));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_horizontal_glyphs_for_colored_floating_square(
                    game.player_pos,
                    game.player_color
                )
        );
    }

    #[test]
    fn test_player_glyphs_when_rounding_to_zero_for_x_and_half_step_up_for_y() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(0.24, 0.26));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_vertical_glyphs_for_colored_floating_square(
                    game.player_pos,
                    game.player_color
                )
        );
    }

    #[test]
    fn test_player_glyphs_when_rounding_to_zero_for_x_and_exactly_half_step_up_for_y() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(0.24, 0.25));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_vertical_glyphs_for_colored_floating_square(
                    game.player_pos,
                    game.player_color
                )
        );
    }

    #[test]
    fn test_player_glyphs_when_rounding_to_zero_for_x_and_exactly_half_step_down_for_y() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(-0.2, -0.25));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_vertical_glyphs_for_colored_floating_square(
                    game.player_pos,
                    game.player_color
                )
        );
    }

    #[test]
    fn test_player_glyphs_when_rounding_to_zero_for_y_and_half_step_right_for_x() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(0.3, 0.1));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_horizontal_glyphs_for_colored_floating_square(
                    game.player_pos,
                    game.player_color
                )
        );
    }

    #[test]
    fn test_player_glyphs_when_rounding_to_zero_for_y_and_half_step_left_for_x() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(-0.3, 0.2));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_horizontal_glyphs_for_colored_floating_square(
                    game.player_pos,
                    game.player_color
                )
        );
    }

    #[test]
    fn test_player_glyphs_for_half_step_up_and_right() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(0.3, 0.4));
        let glyphs = game.get_player_glyphs();

        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1] == None);
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[1][2].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[2][2].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
    }

    #[test]
    fn test_player_glyphs_for_half_step_up_and_left() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(-0.4, 0.26));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[0][2].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[1][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[1][2].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1] == None);
        assert!(glyphs[2][2] == None);
    }

    #[test]
    fn test_player_glyphs_for_half_step_down_and_left() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(-0.26, -0.4999));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[0][0].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[0][1].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
        assert!(glyphs[1][2] == None);
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1] == None);
        assert!(glyphs[2][2] == None);
    }

    #[test]
    fn test_player_glyphs_for_half_step_down_and_right() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(0.26, -0.4));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1] == None);
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[1][2] == None);
        assert!(glyphs[2][0].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[2][1].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
        assert!(glyphs[2][2] == None);
    }

    #[test]
    fn test_off_alignment_player_coarse_rendering_given_diagonal_offset() {
        let mut game = set_up_just_player();

        game.player_pos.add_assign(p(0.4, -0.3));
        game.update_output_buffer();
        let top_left_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos));
        let top_right_glyph = game.get_buffered_glyph(snap_to_grid(game.player_pos + p(1.0, 0.0)));
        let bottom_left_glyph =
            game.get_buffered_glyph(snap_to_grid(game.player_pos + p(0.0, -1.0)));
        let bottom_right_glyph =
            game.get_buffered_glyph(snap_to_grid(game.player_pos + p(1.0, -1.0)));

        assert!(top_left_glyph.character == quarter_block_by_offset((1, -1)));
        assert!(top_right_glyph.character == quarter_block_by_offset((-1, -1)));
        assert!(bottom_left_glyph.character == quarter_block_by_offset((1, 1)));
        assert!(bottom_right_glyph.character == quarter_block_by_offset((-1, 1)));
    }

    #[test]
    // The timing of when to slide to a stop should give the player precision positioning
    fn test_dont_snap_to_grid_when_sliding_to_a_halt() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(0.1, 0.0));
        game.player_vel_bpf = p(0.1, 0.0);
        game.player_acceleration_from_traction = 0.1;
        game.player_desired_direction = p(0, 0);

        game.tick_physics();
        assert!(game.player_vel_bpf.x() == 0.0);
        assert!(offset_from_grid(game.player_pos).x() != 0.0);
    }

    #[ignore]
    #[test]
    // The general case of this is pressing jump any time before landing causing an instant jump when possible
    fn test_allow_early_jump() {}

    #[ignore]
    #[test]
    // The general case of this is allowing a single jump anytime while falling after walking (not jumping!) off a platform
    fn test_allow_late_jump() {}

    #[ignore]
    #[test]
    // This should allow high skill to lead to really fast wall climbing (like in N+)
    fn test_wall_jump_adds_velocity_instead_of_sets_it() {}

    #[ignore]
    #[test]
    // Once we have vertical subsquare positioning up and running, a slow slide down will look cool.
    fn test_slowly_slide_down_when_grabbing_wall() {}

    #[test]
    fn test_dash_no_direction_does_nothing() {
        let mut game = set_up_player_on_platform();
        let start_vel = game.player_vel_bpf;
        game.player_dash();
        assert!(game.player_vel_bpf == start_vel);
    }

    #[test]
    fn test_dash_right_on_ground() {
        let mut game = set_up_player_on_platform();
        let start_vel = game.player_vel_bpf;
        game.player_desired_direction = p(1, 0);
        game.player_dash();
        assert!(
            game.player_vel_bpf == floatify(game.player_desired_direction) * DEFAULT_PLAYER_DASH_V
        );
    }

    #[test]
    fn test_no_passive_horizontal_momentum_loss_while_midair() {
        let mut game = set_up_just_player();
        game.player_desired_direction = p(1, 0);
        let vx = 5.0;
        game.player_vel_bpf = p(vx, 0.0);
        game.tick_physics();

        assert!(game.player_vel_bpf.x() == vx);
    }

    #[test]
    fn test_do_not_grab_wall_while_standing_on_ground() {
        let mut game = set_up_player_on_platform();
        let wall_x = game.player_pos.x() as i32 - 1;
        game.place_line_of_blocks((wall_x, 0), (wall_x, 20), Block::Wall);
        game.player_desired_direction = p(-1, 0);

        assert!(game.player_is_standing_on_block() == true);
        assert!(game.player_is_grabbing_wall() == false);
    }

    #[test]
    fn test_direction_buttons() {
        let mut game = set_up_player_on_platform();
        game.handle_input(Event::Key(Key::Char('a')));
        assert!(game.player_desired_direction == p(-1, 0));
        game.handle_input(Event::Key(Key::Char('s')));
        assert!(game.player_desired_direction == p(0, -1));
        game.handle_input(Event::Key(Key::Char('d')));
        assert!(game.player_desired_direction == p(1, 0));
        game.handle_input(Event::Key(Key::Char('w')));
        assert!(game.player_desired_direction == p(0, 1));

        game.handle_input(Event::Key(Key::Left));
        assert!(game.player_desired_direction == p(-1, 0));
        game.handle_input(Event::Key(Key::Down));
        assert!(game.player_desired_direction == p(0, -1));
        game.handle_input(Event::Key(Key::Right));
        assert!(game.player_desired_direction == p(1, 0));
        game.handle_input(Event::Key(Key::Up));
        assert!(game.player_desired_direction == p(0, 1));
    }

    #[test]
    fn test_different_color_when_go_fast() {
        let mut game = set_up_player_on_platform();
        let stopped_color = game.get_player_glyphs()[1][1].clone().unwrap().fg_color;
        game.player_vel_bpf = p(game.player_max_run_speed_bpf + 5.0, 0.0);
        let fast_color = game.get_player_glyphs()[1][1].clone().unwrap().fg_color;

        assert!(stopped_color != fast_color);
    }

    #[test]
    fn test_draw_visual_braille_line_without_rounding() {
        let mut game = set_up_player_on_platform();
        let start = game.player_pos + p(0.51, 0.1);
        let end = start + p(2.1, 0.1);
        let color = ColorName::Blue;

        // Expected braille:
        // 00 00 00
        // 11 11 10
        // 00 00 00
        // 00 00 00

        game.draw_visual_braille_line(start, end, color);

        assert!(game.get_buffered_glyph(snap_to_grid(start)).fg_color == color);
        assert!(game.get_buffered_glyph(snap_to_grid(start)).character == '\u{2812}');
        assert!(
            game.get_buffered_glyph(snap_to_grid(start) + p(1, 0))
                .fg_color
                == color
        );
        assert!(
            game.get_buffered_glyph(snap_to_grid(start) + p(1, 0))
                .character
                == '\u{2812}' // '⠒'
        );
        assert!(
            game.get_buffered_glyph(snap_to_grid(start) + p(2, 0))
                .fg_color
                == color
        );
        assert!(
            game.get_buffered_glyph(snap_to_grid(start) + p(2, 0))
                .character
                == '\u{2802}'
        );
    }

    #[test]
    fn test_drawn_braille_adds_instead_of_overwrites() {
        let mut game = Game::new(30, 30);
        let start_square = p(12, 5);
        let color = ColorName::Blue;
        let start_upper_line = floatify(start_square) + p(-0.1, 0.1);
        let end_upper_line = floatify(start_square) + p(1.6, 0.1);
        let start_lower_line = floatify(start_square) + p(-0.1, -0.3);
        let end_lower_line = floatify(start_square) + p(1.6, -0.3);

        game.draw_visual_braille_line(start_upper_line, end_upper_line, color);
        game.draw_visual_braille_line(start_lower_line, end_lower_line, color);

        // Expected braille:
        // 00 00 00
        // 11 11 10
        // 00 00 00
        // 11 11 10

        // \u{28D2}, \u{28D2}, \u{2842},

        assert!(game.get_buffered_glyph(start_square + p(0, 0)).character == '\u{28D2}');
        assert!(game.get_buffered_glyph(start_square + p(1, 0)).character == '\u{28D2}');
        assert!(game.get_buffered_glyph(start_square + p(2, 0)).character == '\u{2842}');
    }

    #[test]
    fn test_braille_speed_lines_when_go_fast() {
        let mut game = set_up_player_on_platform_in_box();
        game.player_vel_bpf = p(game.player_color_change_speed_threshold * 5.0, 0.0);
        let start_pos = game.player_pos;
        game.tick_physics();
        game.update_output_buffer();
        for x in 0..3 {
            assert!(Glyph::is_braille(
                game.get_buffered_glyph(snap_to_grid(start_pos) + p(x, 0))
                    .character
            ));
        }
    }

    #[test]
    fn test_recent_player_poses_starts_empty() {
        let mut game = set_up_player_on_platform();
        assert!(game.recent_player_poses.is_empty());
    }

    #[test]
    fn test_recent_player_poses_are_saved() {
        let mut game = set_up_player_on_platform();
        let p0 = game.player_pos;
        let p1 = p(5.0, 2.0);
        let p2 = p(6.7, 3.4);
        game.tick_physics();
        game.player_pos = p1;
        game.tick_physics();
        game.player_pos = p2;
        game.tick_physics();

        assert!(game.recent_player_poses.len() == 3);
        assert!(game.recent_player_poses.get(0).unwrap() == &p2);
        assert!(game.recent_player_poses.get(1).unwrap() == &p1);
        assert!(game.recent_player_poses.get(2).unwrap() == &p0);
    }

    #[test]
    fn test_dash_sets_velocity_rather_than_adds_to_it() {
        let mut game = set_up_just_player();
        game.player_vel_bpf = p(-DEFAULT_PLAYER_DASH_V * 4.0, 0.0);
        game.player_desired_direction = p(1, 0);
        game.player_dash();
        assert!(game.player_vel_bpf.x() > 0.0);
    }

    #[ignore]
    #[test]
    // slomo is cool
    fn test_bullet_time() {}

    #[ignore]
    #[test]
    fn test_speed_lines_do_not_cover_blocks() {}

    #[test]
    fn test_prevent_clip_into_wall_when_colliding_with_internal_block_edge() {
        // data comes from observed bug reproduction
        let start_pos = p(35.599968, 16.5);
        let prev_pos = p(36.899967, 17.25);

        let mut game = Game::new(50, 50);
        game.place_player(start_pos.x(), start_pos.y());
        game.player_vel_bpf = start_pos - prev_pos;
        game.player_desired_direction = p(-1, 0);

        game.place_block(p(34, 17), Block::Wall);
        game.place_block(p(34, 16), Block::Wall);
        game.place_block(p(34, 15), Block::Wall);

        game.num_positions_per_block_to_check_for_collisions = 0.00001;
        game.tick_physics();
        let step = game.player_pos - start_pos;

        assert!(game.player_pos.x() == 35.0);
    }

    #[test]
    fn update_color_threshold_with_jump_delta_v_update() {
        let mut game = set_up_game();
        let start_thresh = game.player_color_change_speed_threshold;
        game.set_player_jump_delta_v(game.player_jump_delta_v + 1.0);
        assert!(game.player_color_change_speed_threshold != start_thresh);
    }

    #[test]
    fn update_color_threshold_with_speed_update() {
        let mut game = set_up_game();
        let start_thresh = game.player_color_change_speed_threshold;
        game.set_player_max_run_speed(game.player_max_run_speed_bpf + 1.0);
        assert!(game.player_color_change_speed_threshold != start_thresh);
    }
}
