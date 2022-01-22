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
use std::fmt::Debug;
use std::io::{stdin, stdout, Write};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant};
use termion::color;
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

// const player_jump_height: i32 = 3;
// const player_jump_hang_frames: i32 = 4;
const MAX_FPS: i32 = 60; // frames per second
const IDEAL_FRAME_DURATION_MS: u128 = (1000.0 / MAX_FPS as f32) as u128;

// a block every two ticks
const PLAYER_DEFAULT_MAX_SPEED_BPS: f32 = 30.0; // blocks per second
const PLAYER_DEFAULT_MAX_SPEED_BPF: f32 = PLAYER_DEFAULT_MAX_SPEED_BPS / MAX_FPS as f32; // blocks per frame
const DEFAULT_PLAYER_JUMP_DELTA_V: f32 = 1.0;
const DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY: f32 = 0.1;
const DEFAULT_PLAYER_ACCELERATION_FROM_TRACTION: f32 = 1.0;
const DEFAULT_PLAYER_COYOTE_TIME_DURATION_S: f32 = 0.2;
const DEFAULT_PLAYER_MAX_COYOTE_FRAMES: i32 =
    ((DEFAULT_PLAYER_COYOTE_TIME_DURATION_S * MAX_FPS as f32) + 1.0) as i32;

// "heighth", "reighth"
const EIGHTH_BLOCKS_FROM_LEFT: &[char] = &[' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];
const EIGHTH_BLOCKS_FROM_BOTTOM: &[char] = &[' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
fn quarter_block_by_offset(half_steps: (i32, i32)) -> char {
    match half_steps {
        (1, -1) => '▗',
        (1, 0) => '▐',
        (1, 1) => '▝',
        (0, -1) => '▄',
        (0, 0) => '█',
        (0, 1) => '▀',
        (-1, -1) => '▖',
        (-1, 0) => '▌',
        (-1, 1) => '▘',
        _ => ' ',
    }
}

fn p<T: 'static>(x: T, y: T) -> Point<T>
where
    T: CoordNum,
{
    return point!(x: x, y: y);
}

// These have no positional information
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Block {
    Air,
    Wall,
    Brick,
    Player,
}

impl Block {
    fn glyph(&self) -> char {
        match self {
            Block::Air => ' ',
            Block::Wall => '█',
            Block::Brick => '▪',
            Block::Player => EIGHTH_BLOCKS_FROM_LEFT[8],
        }
    }

    fn subject_to_block_gravity(&self) -> bool {
        match self {
            Block::Air | Block::Wall | Block::Player => false,
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

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct MovecastCollision {
    pos: Point<i32>,
    normal: Point<i32>,
}

enum ColorName {
    Red,
    Black,
    // White,
    Reset,
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Glyph {
    character: char,
    fg_color: Option<String>,
    bg_color: Option<String>,
}

impl Glyph {
    fn to_string(&self) -> String {
        let mut output = self.character.to_string();
        if let Some(fg_color_string) = self.fg_color.clone() {
            output = format!(
                "{}{}{}",
                fg_color_string,
                output,
                Glyph::fg_color_from_name(ColorName::Reset),
            );
        }
        if let Some(bg_color_string) = self.bg_color.clone() {
            output = format!(
                "{}{}{}",
                bg_color_string,
                output,
                color::Bg(color::Reset).to_string(),
            );
        }
        return output;
    }

    fn from_char(character: char) -> Glyph {
        return Glyph {
            character: character,
            fg_color: None,
            bg_color: None,
        };
    }

    fn fg_color_from_name(color_name: ColorName) -> String {
        match color_name {
            ColorName::Red => color::Fg(color::Red).to_string(),
            // ColorName::White => color::Fg(color::White).to_string(),
            ColorName::Black => color::Fg(color::Black).to_string(),
            ColorName::Reset => color::Fg(color::Reset).to_string(),
        }
    }

    fn bg_color_from_name(color_name: ColorName) -> String {
        match color_name {
            ColorName::Red => color::Bg(color::Red).to_string(),
            // ColorName::White => color::Bg(color::White).to_string(),
            ColorName::Black => color::Bg(color::Black).to_string(),
            ColorName::Reset => color::Bg(color::Reset).to_string(),
        }
    }
    fn red_square_with_horizontal_offset(fraction_of_square_offset: f32) -> Glyph {
        let offset_in_eighths_rounded_towards_inf = (fraction_of_square_offset * 8.0).ceil() as i32;
        assert!(offset_in_eighths_rounded_towards_inf.abs() <= 8);
        if offset_in_eighths_rounded_towards_inf <= 0 {
            return Glyph {
                character: EIGHTH_BLOCKS_FROM_LEFT
                    [(8 + offset_in_eighths_rounded_towards_inf) as usize],
                fg_color: Some(Glyph::fg_color_from_name(ColorName::Red)),
                bg_color: Some(Glyph::bg_color_from_name(ColorName::Black)),
            };
        } else {
            return Glyph {
                character: EIGHTH_BLOCKS_FROM_LEFT[offset_in_eighths_rounded_towards_inf as usize],
                fg_color: Some(Glyph::fg_color_from_name(ColorName::Black)),
                bg_color: Some(Glyph::bg_color_from_name(ColorName::Red)),
            };
        }
    }
    fn red_square_with_vertical_offset(fraction_of_square_offset: f32) -> Glyph {
        let offset_in_eighths_rounded_towards_inf = (fraction_of_square_offset * 8.0).ceil() as i32;
        assert!(offset_in_eighths_rounded_towards_inf.abs() <= 8);
        if offset_in_eighths_rounded_towards_inf <= 0 {
            return Glyph {
                character: EIGHTH_BLOCKS_FROM_BOTTOM
                    [(8 + offset_in_eighths_rounded_towards_inf) as usize],
                fg_color: Some(Glyph::fg_color_from_name(ColorName::Red)),
                bg_color: Some(Glyph::bg_color_from_name(ColorName::Black)),
            };
        } else {
            return Glyph {
                character: EIGHTH_BLOCKS_FROM_BOTTOM
                    [(offset_in_eighths_rounded_towards_inf) as usize],
                fg_color: Some(Glyph::fg_color_from_name(ColorName::Black)),
                bg_color: Some(Glyph::bg_color_from_name(ColorName::Red)),
            };
        }
    }
    fn red_square_with_half_step_offset(offset: Point<f32>) -> Glyph {
        let step = trunc(offset * 4.0);
        Glyph {
            character: quarter_block_by_offset((step.x(), step.y())),
            fg_color: Some(Glyph::fg_color_from_name(ColorName::Red)),
            bg_color: Some(Glyph::bg_color_from_name(ColorName::Black)),
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
    player_max_run_speed_bpf: f32,
    player_vel_bpf: Point<f32>,
    player_desired_x_direction: i32,
    player_jump_delta_v: f32,
    player_acceleration_from_gravity: f32,
    player_acceleration_from_traction: f32,
    player_remaining_coyote_frames: i32,
    player_max_coyote_frames: i32,
}

impl Game {
    fn new(width: u16, height: u16) -> Game {
        Game {
            grid: vec![vec![Block::Air; height as usize]; width as usize],
            output_buffer: vec![vec![Glyph::from_char(' '); height as usize]; width as usize],
            output_on_screen: vec![vec![Glyph::from_char(' '); height as usize]; width as usize],
            terminal_size: (width, height),
            prev_mouse_pos: (1, 1),
            running: true,
            selected_block: Block::Wall,
            player_alive: false,
            player_pos: p(0.0, 0.0),
            player_max_run_speed_bpf: PLAYER_DEFAULT_MAX_SPEED_BPF,
            player_vel_bpf: Point::<f32>::new(0.0, 0.0),
            player_desired_x_direction: 0,
            player_jump_delta_v: DEFAULT_PLAYER_JUMP_DELTA_V,
            player_acceleration_from_gravity: DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY,
            player_acceleration_from_traction: DEFAULT_PLAYER_ACCELERATION_FROM_TRACTION,
            player_remaining_coyote_frames: DEFAULT_PLAYER_MAX_COYOTE_FRAMES,
            player_max_coyote_frames: DEFAULT_PLAYER_MAX_COYOTE_FRAMES,
        }
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
    fn snap_to_grid(world_pos: Point<f32>) -> Point<i32> {
        return round(world_pos);
    }
    fn offset_from_grid(world_pos: Point<f32>) -> Point<f32> {
        return world_pos - floatify(Game::snap_to_grid(world_pos));
    }
    fn get_block(&self, pos: Point<i32>) -> Block {
        return self.grid[pos.x() as usize][pos.y() as usize];
    }
    fn set_block(&mut self, pos: Point<i32>, block: Block) {
        self.grid[pos.x() as usize][pos.y() as usize] = block;
    }

    fn draw_line(&mut self, pos0: (i32, i32), pos1: (i32, i32), block: Block) {
        for (x1, y1) in line_drawing::Bresenham::new(pos0, pos1) {
            self.grid[x1 as usize][y1 as usize] = block;
        }
    }

    fn draw_point(&mut self, pos: Point<i32>, block: Block) {
        self.grid[pos.x() as usize][pos.y() as usize] = block;
    }

    fn clear(&mut self) {
        let (width, height) = termion::terminal_size().unwrap();
        self.grid = vec![vec![Block::Air; height as usize]; width as usize];
    }

    fn place_player(&mut self, x: f32, y: f32) {
        if self.player_alive {
            self.kill_player();
        }
        let grid_point = Game::snap_to_grid(p(x, y));
        self.grid[grid_point.x() as usize][grid_point.y() as usize] = Block::Player;
        self.player_vel_bpf = p(0.0, 0.0);
        self.player_desired_x_direction = 0;
        self.player_pos = p(x, y);
        self.player_alive = true;
        self.player_remaining_coyote_frames = 0;
    }
    fn player_jump(&mut self) {
        if self.player_is_grabbing_wall() {
            self.player_vel_bpf.add_assign(p(
                self.player_jump_delta_v * -self.player_wall_grab_direction() as f32,
                0.0,
            ));
            self.player_desired_x_direction *= -1;
        }
        self.player_vel_bpf.set_y(self.player_jump_delta_v);
    }
    fn player_jump_if_possible(&mut self) {
        if self.player_is_supported() || self.player_is_grabbing_wall() {
            self.player_jump();
        }
    }

    fn player_set_desired_x_direction(&mut self, new_x_dir: i32) {
        if new_x_dir != self.player_desired_x_direction {
            self.player_desired_x_direction = new_x_dir.sign();
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
                Key::Char('a') | Key::Left => self.player_set_desired_x_direction(-1),
                Key::Char('s') | Key::Down => self.player_set_desired_x_direction(0),
                Key::Char('d') | Key::Right => self.player_set_desired_x_direction(1),
                _ => {}
            },
            Event::Mouse(me) => match me {
                MouseEvent::Press(MouseButton::Left, term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.draw_point(p(x, y), self.selected_block);
                    self.prev_mouse_pos = (x, y);
                }
                MouseEvent::Press(MouseButton::Right, term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.place_player(x as f32, y as f32);
                }
                MouseEvent::Hold(term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.draw_line(self.prev_mouse_pos, (x, y), self.selected_block);
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

        let player_glyphs = self.get_player_glyphs();
        let grid_pos = Game::snap_to_grid(self.player_pos);

        for i in 0..player_glyphs.len() {
            for j in 0..player_glyphs[i].len() {
                if let Some(glyph) = player_glyphs[i][j].clone() {
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
        let width = 3;
        let mut output = vec![vec![None; width]; width];

        let c = width / 2 as usize;

        let grid_offset = Game::offset_from_grid(self.player_pos);
        let x_offset = grid_offset.x();
        let y_offset = grid_offset.y();
        let offset_dir = round(sign(grid_offset));
        if y_offset == 0.0 {
            for i in 0..3 {
                let x = i as i32 - 1;
                if !(-offset_dir.x() == x) {
                    output[i][c] = Some(Glyph::red_square_with_horizontal_offset(
                        x_offset - x as f32,
                    ));
                }
            }
        } else if x_offset == 0.0 {
            for j in 0..3 {
                let y = j as i32 - 1;
                if !(-offset_dir.y() == y) {
                    output[c][j] =
                        Some(Glyph::red_square_with_vertical_offset(y_offset - y as f32));
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    let x = i as i32 - 1;
                    let y = j as i32 - 1;
                    let square = p(x as f32, y as f32);
                    if !(-offset_dir.x() == x) && !(-offset_dir.y() == y) {
                        let glyph = Glyph::red_square_with_half_step_offset(
                            Game::offset_from_grid(self.player_pos) - square,
                        );
                        if glyph.character != ' ' {
                            output[i][j] = Some(glyph);
                        }
                    }
                }
            }
        }
        return output;
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

    fn move_player_to(&mut self, pos: Point<f32>) {
        self.set_block(Game::snap_to_grid(self.player_pos), Block::Air);
        self.set_block(Game::snap_to_grid(pos), Block::Player);
        self.player_pos = pos;
    }

    fn kill_player(&mut self) {
        self.set_block(Game::snap_to_grid(self.player_pos), Block::Air);
        self.player_alive = false;
    }

    fn player_is_grabbing_wall(&self) -> bool {
        if self.player_desired_x_direction != 0 && self.player_vel_bpf.y() <= 0.0 {
            if let Some(block) =
                self.get_block_relative_to_player(p(self.player_desired_x_direction.sign(), 0))
            {
                return block.wall_grabbable();
            }
        }
        return false;
    }
    fn player_wall_grab_direction(&self) -> i32 {
        // TODO: is this good?
        if self.player_is_grabbing_wall() {
            return self.player_desired_x_direction.sign();
        } else {
            return 0;
        }
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
        let desired_acceleration_direction = self.player_desired_x_direction.sign();

        let trying_to_stop = desired_acceleration_direction == 0 && start_x_vel != 0.0;
        let started_above_max_speed = start_x_vel.abs() > self.player_max_run_speed_bpf;

        let real_acceleration_direction;
        if trying_to_stop || started_above_max_speed {
            real_acceleration_direction = -start_x_vel.sign() as i32;
        } else {
            real_acceleration_direction = desired_acceleration_direction;
        }
        let delta_vx =
            real_acceleration_direction.sign() as f32 * self.player_acceleration_from_traction;
        let mut end_x_vel = self.player_vel_bpf.x() + delta_vx;
        let changed_direction = start_x_vel * end_x_vel < 0.0;

        if trying_to_stop && changed_direction {
            end_x_vel = 0.0;
        }

        let want_to_go_faster = start_x_vel.sign() == desired_acceleration_direction.sign() as f32;
        let ended_below_max_speed = end_x_vel.abs() < self.player_max_run_speed_bpf;
        if started_above_max_speed && ended_below_max_speed && want_to_go_faster {
            end_x_vel = end_x_vel.sign() * self.player_max_run_speed_bpf;
        }
        // if want go fast, but starting less than max speed.  Can't go past max speed.
        if !started_above_max_speed && !ended_below_max_speed {
            end_x_vel = end_x_vel.sign() * self.player_max_run_speed_bpf;
        }

        if end_x_vel == 0.0 {
            self.player_pos
                .set_x(Game::snap_to_grid(self.player_pos).x() as f32); // TODO: double check this
        }
        self.player_vel_bpf.set_x(end_x_vel);
    }

    fn apply_player_acceleration_from_gravity(&mut self) {
        self.player_vel_bpf
            .add_assign(p(0.0, -self.player_acceleration_from_gravity));
    }

    fn apply_player_motion(&mut self) {
        self.update_player_acceleration();

        let ideal_step: Point<f32> = self.player_vel_bpf;
        let ideal_new_pos = self.player_pos + ideal_step;
        let start_square = Game::snap_to_grid(self.player_pos);

        let collision_checks_per_block_travelled = 8.0;
        let num_points_to_check =
            (magnitude(ideal_step) * collision_checks_per_block_travelled).floor() as i32;
        let mut prev_point_to_check = self.player_pos.clone();
        let mut early_stop_point: Option<Point<f32>> = None;
        'outer: for i in 0..num_points_to_check {
            let point_to_check = self.player_pos
                + direction(ideal_step) * (i as f32 / collision_checks_per_block_travelled);

            for touching_grid_point in
                Game::grid_squares_overlapped_by_floating_unit_square(point_to_check)
            {
                if self.in_world(touching_grid_point)
                    && self.get_block(touching_grid_point) == Block::Wall
                {
                    // snap player to just before collision
                    early_stop_point = Some(prev_point_to_check);

                    break 'outer;
                }
            }
            prev_point_to_check = point_to_check;
        }
        let actual_endpoint: Point<f32>;
        if let Some(point) = early_stop_point {
            actual_endpoint = point;
            self.player_vel_bpf = p(0.0, 0.0);
            // TODO: kill velocity along only the one axis
        } else {
            actual_endpoint = ideal_new_pos;
        }

        let step_taken: Point<f32>;

        if !self.in_world(Game::snap_to_grid(actual_endpoint)) {
            // Player went out of bounds and died
            self.kill_player();
            return;
        } else {
            // no collision, and in world
            step_taken = actual_endpoint - self.player_pos;
            self.move_player_to(actual_endpoint);
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

    fn grid_squares_overlapped_by_floating_unit_square(pos: Point<f32>) -> Vec<Point<i32>> {
        let mut output = Vec::<Point<i32>>::new();
        let offset_direction = round(sign(Game::offset_from_grid(pos)));
        for i in 0..3 {
            for j in 0..3 {
                let square = p(i as i32 - 1, j as i32 - 1);
                if square == offset_direction {
                    output.push(Game::snap_to_grid(pos) + square);
                }
            }
        }
        return output;
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
    fn movecast(&self, start: Point<i32>, end: Point<i32>) -> Option<MovecastCollision> {
        let mut prev_pos = start;
        for (x, y) in line_drawing::WalkGrid::new((start.x(), start.y()), (end.x(), end.y())) {
            let pos = p(x, y);
            if self.in_world(pos) {
                if self.get_block(pos) != Block::Air && self.get_block(pos) != Block::Player {
                    // one before the block
                    return Some(MovecastCollision {
                        pos: prev_pos,
                        normal: prev_pos - pos,
                    });
                } else if pos == end {
                    // if we made it to the end, no collision
                    return None;
                }
            } else {
                // ran off the grid.  No collision
                return None;
            }
            // the prev_pos will be the start pos for the first 2 run throughs of the loop
            prev_pos = pos;
        }
        // shouldn't get here
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
        let target_pos = Game::snap_to_grid(self.player_pos) + rel_pos;
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
        self.player_jump_delta_v = 0.5;
        self.player_acceleration_from_gravity = 0.01;
        self.player_acceleration_from_traction = 0.1;
        self.player_max_run_speed_bpf = 0.3;

        let bottom_left = (
            (self.terminal_size.0 / 5) as i32,
            (self.terminal_size.1 / 4) as i32,
        );
        self.draw_line(
            bottom_left,
            ((4 * self.terminal_size.0 / 5) as i32, bottom_left.1),
            Block::Wall,
        );
        self.draw_line(
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

fn backup_point_from_unit_square_collision(
    start_point: Point<f32>,
    step_taken: Point<f32>,
    grid_square_center: Point<i32>,
) -> Point<f32>
where
{
    // formulates the problem as a point crossing the boundary of an r=1 square
    let movement_line = geo::Line::new(start_point, start_point + step_taken);
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
    

    // four intersections with extended walls of stationary square
    candidate_edge_intersections.sort_by(|a, b| start_point.euclidean_distance(a).partial_cmp(&start_point.euclidean_distance(b)).unwrap());
    if candidate_edge_intersections.len() < 1 {
        panic!("Failed to find intersection");
    }
    return candidate_edge_intersections[0];

}

fn e<T: CoordNum + num::Signed + std::fmt::Display>(dir_num: T) -> Point<T> {
    let dir_num_int = dir_num.to_i32().unwrap() % 4;
    match dir_num_int {
        0 => Point::<T>::new(T::one(), T::zero()),
        1 => Point::<T>::new(T::zero(), T::one()),
        2 => Point::<T>::new(-T::one(), T::zero()),
        3 => Point::<T>::new(T::zero(), -T::one()),
        _ => panic!("bad direction number: {}", dir_num),
    }
}

fn trunc(vec: Point<f32>) -> Point<i32> {
    return Point::<i32>::new(vec.x().trunc() as i32, vec.y().trunc() as i32);
}

fn round(vec: Point<f32>) -> Point<i32> {
    return Point::<i32>::new(vec.x().round() as i32, vec.y().round() as i32);
}
fn floatify(vec: Point<i32>) -> Point<f32> {
    return Point::<f32>::new(vec.x() as f32, vec.y() as f32);
}

fn magnitude(vec: Point<f32>) -> f32 {
    return (vec.x().pow(2.0) + vec.y().pow(2.0)).sqrt();
}
fn direction(vec: Point<f32>) -> Point<f32> {
    return vec / magnitude(vec);
}

fn fract(vec: Point<f32>) -> Point<f32> {
    return Point::<f32>::new(vec.x().fract(), vec.y().fract());
}

fn sign<T>(p: Point<T>) -> Point<T>
where
    T: SignedExt + CoordNum,
{
    return Point::<T>::new(p.x().sign(), p.y().sign());
}

trait SignedExt: num::Signed {
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

trait PointExt {
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

    fn set_up_just_player() -> Game {
        let mut game = Game::new(30, 30);
        game.place_player(15.0, 11.0);
        return game;
    }

    fn set_up_player_on_platform() -> Game {
        let mut game = set_up_just_player();
        game.draw_line((10, 10), (20, 10), Block::Wall);
        return game;
    }

    fn set_up_player_hanging_on_wall_on_left() -> Game {
        let mut game = set_up_just_player();
        game.draw_line((14, 0), (14, 20), Block::Wall);
        game.player_desired_x_direction = -1;
        return game;
    }

    #[test]
    fn test_placement_and_gravity() {
        let mut game = Game::new(30, 30);
        game.draw_point(p(10, 10), Block::Wall);
        game.draw_point(p(20, 10), Block::Brick);

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

        assert!(game.grid[x1 as usize][y1 as usize] == Block::Player);
        assert!(game.player_pos == p(x1, y1));
        assert!(game.player_alive == true);

        game.place_player(x2, y2);
        assert!(game.grid[x1 as usize][y1 as usize] == Block::Air);
        assert!(game.grid[x2 as usize][y2 as usize] == Block::Player);
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
    fn test_movecast() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);

        assert!(
            game.movecast(p(15, 9), p(15, 11))
                == Some(MovecastCollision {
                    pos: p(15, 9),
                    normal: p(0, -1)
                })
        );
        assert!(
            game.movecast(p(15, 10), p(15, 11))
                == Some(MovecastCollision {
                    pos: p(15, 10),
                    normal: p(0, 0)
                })
        );
        assert!(
            game.movecast(p(15, 9), p(17, 11))
                == Some(MovecastCollision {
                    pos: p(15, 9),
                    normal: p(0, -1)
                })
        );
        assert!(
            game.movecast(p(15, 9), p(17, 110))
                == Some(MovecastCollision {
                    pos: p(15, 9),
                    normal: p(0, -1)
                })
        );
        assert!(game.movecast(p(150, 9), p(17, 11)) == None);
        assert!(game.movecast(p(1, 9), p(-17, 9)) == None);
        assert!(game.movecast(p(15, 9), p(17, -11)) == None);
    }

    #[test]
    fn test_movecast_ignore_player() {
        let mut game = set_up_player_on_platform();

        assert!(game.movecast(p(15, 11), p(15, 13)) == None);
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
        game.player_desired_x_direction = 1;

        game.tick_physics();

        assert!(game.player_pos == p(16.0, 11.0));
        assert!(game.grid[15][11] == Block::Air);
        assert!(game.grid[16][11] == Block::Player);

        game.place_player(15.0, 11.0);
        assert!(game.player_desired_x_direction == 0);
        game.player_desired_x_direction = -1;

        game.tick_physics();
        game.tick_physics();

        assert!(game.grid[15][11] == Block::Air);
        assert!(game.grid[13][11] == Block::Player);
        assert!(game.player_pos == p(13.0, 11.0));
    }
    #[test]
    fn test_stop_on_collision() {
        let mut game = set_up_player_on_platform();
        game.draw_point(round(game.player_pos) + p(1, 0), Block::Wall);
        game.player_max_run_speed_bpf = 1.0;
        game.player_desired_x_direction = 1;

        game.tick_physics();

        assert!(game.player_pos == p(15.0, 11.0));
        assert!(game.grid[16][11] == Block::Wall);
        assert!(game.grid[15][11] == Block::Player);
        assert!(game.player_vel_bpf.x() == 0.0);
    }

    #[test]
    fn test_snap_to_object_on_collision() {
        let mut game = set_up_player_on_platform();
        game.draw_point(round(game.player_pos) + p(2, 0), Block::Wall);
        game.player_pos.add_assign(p(0.999, 0.0));
        game.player_max_run_speed_bpf = 1.0;
        game.player_acceleration_from_traction = 999.0;
        game.player_desired_x_direction = 1;

        game.tick_physics();

        assert!(game.player_pos == p(16.0, 11.0));
        assert!(game.grid[17][11] == Block::Wall);
    }

    #[test]
    fn test_move_player_slowly() {
        let mut game = set_up_player_on_platform();
        game.player_max_run_speed_bpf = 0.49;
        game.player_acceleration_from_traction = 999.9; // rilly fast
        game.player_desired_x_direction = 1;

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
        game.player_desired_x_direction = 1;

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
        game.draw_point(p(16, 11), Block::Wall);
        game.player_max_run_speed_bpf = 2.0;
        game.player_desired_x_direction = 1;

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
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.draw_line((14, 10), (14, 20), Block::Wall);

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
        game.player_desired_x_direction = 1;
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
    fn test_wall_jump() {
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
    #[ignore]
    #[test]
    fn test_wall_jump_while_sliding_up_wall() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_vel_bpf.set_y(1.0);
        game.player_remaining_coyote_frames = 0;
        game.player_jump_if_possible();
        assert!(game.player_vel_bpf.x() > 0.0);
    }

    #[test]
    fn test_draw_to_output_buffer() {
        let mut game = set_up_player_on_platform();
        game.update_output_buffer();
        assert!(
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos))
                .character
                == EIGHTH_BLOCKS_FROM_LEFT[8]
        );
        assert!(
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(0.0, -1.0)))
                .character
                == Block::Wall.glyph()
        );
    }

    #[test]
    fn test_horizontal_sub_glyph_positioning_on_right_rounding_up() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(0.5, 0.0));
        game.update_output_buffer();
        let left_glyph = game.get_buffered_glyph(Game::snap_to_grid(game.player_pos));
        let right_glyph =
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(1.0, 0.0)));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[4]);
        assert!(left_glyph.fg_color == Some(Glyph::fg_color_from_name(ColorName::Black)));
        assert!(left_glyph.bg_color == Some(Glyph::bg_color_from_name(ColorName::Red)));
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[4]);
        assert!(right_glyph.fg_color == Some(Glyph::fg_color_from_name(ColorName::Red)));
        assert!(right_glyph.bg_color == Some(Glyph::bg_color_from_name(ColorName::Black)));
    }

    #[test]
    fn test_horizontal_sub_glyph_positioning_on_right_below_rounding_point() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(0.49, 0.0));
        game.update_output_buffer();
        let left_glyph = game.get_buffered_glyph(Game::snap_to_grid(game.player_pos));
        let right_glyph =
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(1.0, 0.0)));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[4]);
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[4]);
    }
    #[test]
    fn test_horizontal_sub_glyph_positioning_on_left() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(-0.2, 0.0));
        game.update_output_buffer();

        let left_glyph =
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(-1.0, 0.0)));
        let right_glyph = game.get_buffered_glyph(Game::snap_to_grid(game.player_pos));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[7]);
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[7]);
    }
    #[test]
    fn test_horizontal_sub_glyph_positioning_on_left_above_rounding_point() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(-0.49, 0.0));
        game.update_output_buffer();

        let left_glyph =
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(-1.0, 0.0)));
        let right_glyph = game.get_buffered_glyph(Game::snap_to_grid(game.player_pos));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[5]);
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[5]);
    }
    #[test]
    fn test_horizontal_sub_glyph_positioning_on_left_rounding_down() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(-0.5, 0.0));
        game.update_output_buffer();

        let left_glyph =
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(-1.0, 0.0)));
        let right_glyph = game.get_buffered_glyph(Game::snap_to_grid(game.player_pos));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[4]);
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[4]);
    }
    #[test]
    fn test_vertical_sub_glyph_positioning_upwards() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(0.0, 0.5));
        game.update_output_buffer();

        let top_glyph = game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(0.0, 1.0)));
        let bottom_glyph = game.get_buffered_glyph(Game::snap_to_grid(game.player_pos));
        assert!(top_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[4]);
        assert!(top_glyph.fg_color == Some(Glyph::fg_color_from_name(ColorName::Red)));
        assert!(top_glyph.bg_color == Some(Glyph::bg_color_from_name(ColorName::Black)));
        assert!(bottom_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[4]);
        assert!(bottom_glyph.fg_color == Some(Glyph::fg_color_from_name(ColorName::Black)));
        assert!(bottom_glyph.bg_color == Some(Glyph::bg_color_from_name(ColorName::Red)));
    }
    #[test]
    fn test_vertical_sub_glyph_positioning_downwards() {
        let mut game = set_up_player_on_platform();
        game.player_pos.add_assign(p(0.0, -0.2));
        game.update_output_buffer();

        let top_glyph = game.get_buffered_glyph(Game::snap_to_grid(game.player_pos));
        let bottom_glyph =
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(0.0, -1.0)));
        assert!(top_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[7]);
        assert!(top_glyph.fg_color == Some(Glyph::fg_color_from_name(ColorName::Red)));
        assert!(top_glyph.bg_color == Some(Glyph::bg_color_from_name(ColorName::Black)));
        assert!(bottom_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[7]);
        assert!(bottom_glyph.fg_color == Some(Glyph::fg_color_from_name(ColorName::Black)));
        assert!(bottom_glyph.bg_color == Some(Glyph::bg_color_from_name(ColorName::Red)));
    }
    #[test]
    fn test_red_square_with_half_step_offsets() {
        assert!(
            Glyph::red_square_with_half_step_offset(p(0.0, 0.0)).character
                == quarter_block_by_offset((0, 0))
        );
        assert!(
            Glyph::red_square_with_half_step_offset(p(0.0, 0.0)).fg_color
                == Some(Glyph::fg_color_from_name(ColorName::Red))
        );
        assert!(
            Glyph::red_square_with_half_step_offset(p(0.0, 0.0)).bg_color
                == Some(Glyph::bg_color_from_name(ColorName::Black))
        );
        assert!(
            Glyph::red_square_with_half_step_offset(p(0.1, 0.1)).character
                == quarter_block_by_offset((0, 0))
        );
        assert!(
            Glyph::red_square_with_half_step_offset(p(0.24, 0.0)).character
                == quarter_block_by_offset((0, 0))
        );
        assert!(
            Glyph::red_square_with_half_step_offset(p(0.2, 0.4)).character
                == quarter_block_by_offset((0, 1))
        );
        assert!(
            Glyph::red_square_with_half_step_offset(p(-0.499, 0.4)).character
                == quarter_block_by_offset((-1, 1))
        );
        assert!(
            Glyph::red_square_with_half_step_offset(p(0.5, 0.0)).character
                == quarter_block_by_offset((2, 0))
        );
    }
    #[test]
    fn test_player_glyph_when_rounding_to_zero_for_both_axes() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(0.1, 0.1));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[2][2] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((0, 0)));
    }
    #[test]
    fn test_player_glyphs_when_rounding_to_zero_for_x_and_half_step_for_y() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(0.3, 0.1));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[2][2] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((1, 0)));
        assert!(glyphs[1][2].clone().unwrap().character == quarter_block_by_offset((-1, 0)));
    }

    #[test]
    fn test_player_glyphs_for_half_step_up_and_right() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(0.3, 0.4));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[0][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[1][2].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[2][1].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[2][2].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
    }

    #[test]
    fn test_player_glyphs_for_half_step_down_and_left() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(-0.26, -0.4999));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[2][1] == None);
        assert!(glyphs[0][0].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[0][1].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[1][0].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
    }

    #[test]
    fn test_player_glyphs_for_half_step_down_and_right() {
        let mut game = set_up_just_player();
        game.player_pos.add_assign(p(0.26, -0.499));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[0][1] == None);
        assert!(glyphs[1][0].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[2][0].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[2][1].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
    }
    #[ignore]
    #[test]
    // When jumping at an angle, can't use the high precision sub-square glyphs, so fall back to the half-grid precision ones
    fn test_off_alignment_player_coarse_rendering_given_slight_offset() {
        let mut game = set_up_player_on_platform();

        game.player_pos.add_assign(p(0.1, 0.6));
        game.update_output_buffer();
        let player_half_step = p(0, 1);
        let top_glyph = game.get_buffered_glyph(Game::snap_to_grid(
            game.player_pos + floatify(player_half_step),
        ));
        let bottom_glyph = game.get_buffered_glyph(Game::snap_to_grid(game.player_pos));
        assert!(top_glyph.character == quarter_block_by_offset((-player_half_step).x_y()));
        assert!(top_glyph.fg_color == Some(Glyph::fg_color_from_name(ColorName::Red)));
        assert!(top_glyph.bg_color == Some(Glyph::bg_color_from_name(ColorName::Black)));
        assert!(bottom_glyph.character == quarter_block_by_offset((player_half_step).x_y()));
        assert!(bottom_glyph.fg_color == Some(Glyph::fg_color_from_name(ColorName::Red)));
        assert!(bottom_glyph.bg_color == Some(Glyph::bg_color_from_name(ColorName::Black)));
    }
    #[ignore]
    #[test]
    fn test_off_alignment_player_coarse_rendering_given_diagonal_offset() {
        let mut game = set_up_just_player();

        game.player_pos.add_assign(p(0.8, -0.6));
        game.update_output_buffer();
        let top_left_glyph = game.get_buffered_glyph(Game::snap_to_grid(game.player_pos));
        let top_right_glyph =
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(1.0, 0.0)));
        let bottom_left_glyph =
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(0.0, -1.0)));
        let bottom_right_glyph =
            game.get_buffered_glyph(Game::snap_to_grid(game.player_pos + p(1.0, -1.0)));

        assert!(top_left_glyph.character == quarter_block_by_offset((1, -1)));
        assert!(top_right_glyph.character == quarter_block_by_offset((-1, -1)));
        assert!(bottom_left_glyph.character == quarter_block_by_offset((1, 1)));
        assert!(bottom_right_glyph.character == quarter_block_by_offset((-1, 1)));
    }
    #[test]
    fn overlap_detection_rounds_down() {
        let point = p(57.0, -90.0);
        let squares = Game::grid_squares_overlapped_by_floating_unit_square(point);
        assert!(squares.len() == 1);
        assert!(squares[0] == Game::snap_to_grid(point));
    }

    #[test]
    fn test_offset_from_grid_rounds_to_zero() {
        assert!(Game::offset_from_grid(p(9.0, -9.0)) == p(0.0, 0.0));
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
        assert!(sign(Game::offset_from_grid(p(9.0, -9.0))) == p(0.0, 0.0));
    }
    #[test]
    fn test_snap_to_grid_at_zero() {
        assert!(Game::snap_to_grid(p(0.0, 0.0)) == p(0, 0));
    }

    #[test]
    fn test_snap_to_grid_rounding_down_from_positive_x() {
        assert!(Game::snap_to_grid(p(0.4, 0.0)) == p(0, 0));
    }

    #[test]
    fn test_snap_to_grid_rounding_up_diagonally() {
        assert!(Game::snap_to_grid(p(0.9, 59.51)) == p(1, 60));
    }

    #[test]
    fn test_snap_to_grid_rounding_up_diagonally_in_the_negazone() {
        assert!(Game::snap_to_grid(p(-0.9, -59.51)) == p(-1, -60));
    }

    #[ignore]
    #[test]
    // The timing of when to slide to a stop should give the player precision positioning
    fn test_dont_snap_to_grid_when_sliding_to_a_halt() {}
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
    fn test_collision_point_head_on_horizontal() {
        let start_point = p(0.0, 0.0);
        let step = p(3.0, 0.0);
        let block_center = p(3, 0);
        assert!(
            backup_point_from_unit_square_collision(start_point, step, block_center)
                == p(2.0, 0.0)
        );
    }

    #[test]
    fn test_collision_point_head_slightly_offset_from_vertical() {
        let start_point = p(0.3, 0.0);
        let step = p(0.0, 5.0);
        let block_center = p(0, 5);
        assert!(
            backup_point_from_unit_square_collision(start_point, step, block_center)
                == p(start_point.x(), 4.0)
        );
    }

    #[test]
    fn test_collision_point_slightly_diagonalish() {
        let start_point = p(5.0, 0.0);
        let step = p(3.0, 3.0);
        let block_center = p(7, 1);
        assert!(
            backup_point_from_unit_square_collision(start_point, step, block_center)
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
}
