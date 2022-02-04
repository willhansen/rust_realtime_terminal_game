extern crate geo;

use termion::color;

use geo::{point, CoordNum, Point};
use std::cmp::min;

use crate::utility::*;

// "heighth", "reighth"
pub const EIGHTH_BLOCKS_FROM_LEFT: &[char] = &[' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];
pub const EIGHTH_BLOCKS_FROM_BOTTOM: &[char] = &[' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

pub type BrailleArray = [[bool; 4]; 2];

pub fn quarter_block_by_offset(half_steps: (i32, i32)) -> char {
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
#[derive(Clone, PartialEq, Eq, Debug, Copy)]
pub enum ColorName {
    Red,
    Green,
    Blue,
    Black,
    White,
    Reset,
}

#[derive(Clone, PartialEq, Eq, Debug, Copy)]
pub struct Glyph {
    pub character: char,
    pub fg_color: ColorName,
    pub bg_color: ColorName,
}

impl Glyph {
    pub fn to_string(&self) -> String {
        let mut output = self.character.to_string();
        //if self.fg_color != ColorName::White {
        output = format!(
            "{}{}{}",
            Glyph::fg_color_from_name(self.fg_color),
            output,
            Glyph::fg_color_from_name(ColorName::White),
        );
        //}
        //if self.bg_color != ColorName::Black {
        output = format!(
            "{}{}{}",
            Glyph::bg_color_from_name(self.bg_color),
            output,
            Glyph::bg_color_from_name(ColorName::Black),
        );
        //}
        return output;
    }

    pub fn from_char(character: char) -> Glyph {
        Glyph {
            character,
            fg_color: ColorName::White,
            bg_color: ColorName::Black,
        }
    }

    pub fn fg_color_from_name(color_name: ColorName) -> String {
        match color_name {
            ColorName::Red => color::Fg(color::Red).to_string(),
            ColorName::Green => color::Fg(color::Green).to_string(),
            ColorName::Blue => color::Fg(color::Blue).to_string(),
            ColorName::White => color::Fg(color::White).to_string(),
            ColorName::Black => color::Fg(color::Black).to_string(),
            ColorName::Reset => color::Fg(color::Reset).to_string(),
        }
    }

    pub fn bg_color_from_name(color_name: ColorName) -> String {
        match color_name {
            ColorName::Red => color::Bg(color::Red).to_string(),
            ColorName::Green => color::Bg(color::Green).to_string(),
            ColorName::Blue => color::Bg(color::Blue).to_string(),
            ColorName::White => color::Bg(color::White).to_string(),
            ColorName::Black => color::Bg(color::Black).to_string(),
            ColorName::Reset => color::Bg(color::Reset).to_string(),
        }
    }

    pub fn square_with_horizontal_offset(fraction_of_square_offset: f32) -> Glyph {
        Glyph::colored_square_with_horizontal_offset(fraction_of_square_offset, ColorName::White)
    }

    pub fn colored_square_with_horizontal_offset(
        fraction_of_square_offset: f32,
        color_name: ColorName,
    ) -> Glyph {
        let offset_in_eighths_rounded_towards_inf = (fraction_of_square_offset * 8.0).ceil() as i32;
        assert!(offset_in_eighths_rounded_towards_inf.abs() <= 8);
        if offset_in_eighths_rounded_towards_inf <= 0 {
            return Glyph {
                character: EIGHTH_BLOCKS_FROM_LEFT
                    [(8 + offset_in_eighths_rounded_towards_inf) as usize],
                fg_color: color_name,
                bg_color: ColorName::Black,
            };
        } else {
            return Glyph {
                character: EIGHTH_BLOCKS_FROM_LEFT[offset_in_eighths_rounded_towards_inf as usize],
                fg_color: ColorName::Black,
                bg_color: color_name,
            };
        }
    }
    pub fn square_with_vertical_offset(fraction_of_square_offset: f32) -> Glyph {
        return Glyph::colored_square_with_vertical_offset(
            fraction_of_square_offset,
            ColorName::White,
        );
    }
    pub fn colored_square_with_vertical_offset(
        fraction_of_square_offset: f32,
        color_name: ColorName,
    ) -> Glyph {
        let offset_in_eighths_rounded_towards_inf = (fraction_of_square_offset * 8.0).ceil() as i32;
        assert!(offset_in_eighths_rounded_towards_inf.abs() <= 8);
        if offset_in_eighths_rounded_towards_inf <= 0 {
            return Glyph {
                character: EIGHTH_BLOCKS_FROM_BOTTOM
                    [(8 + offset_in_eighths_rounded_towards_inf) as usize],
                fg_color: color_name,
                bg_color: ColorName::Black,
            };
        } else {
            return Glyph {
                character: EIGHTH_BLOCKS_FROM_BOTTOM
                    [(offset_in_eighths_rounded_towards_inf) as usize],
                fg_color: ColorName::Black,
                bg_color: color_name,
            };
        }
    }
    pub fn colored_square_with_half_step_offset(
        offset: Point<f32>,
        color_name: ColorName,
    ) -> Glyph {
        let step = round_vector_with_tie_break_toward_inf(offset * 2.0);
        Glyph {
            character: quarter_block_by_offset((step.x(), step.y())),
            fg_color: color_name,
            bg_color: ColorName::Black,
        }
    }

    pub fn get_glyphs_for_floating_square(pos: Point<f32>) -> Vec<Vec<Option<Glyph>>> {
        Glyph::get_glyphs_for_colored_floating_square(pos, ColorName::White)
    }

    pub fn get_glyphs_for_colored_floating_square(
        pos: Point<f32>,
        color: ColorName,
    ) -> Vec<Vec<Option<Glyph>>> {
        let grid_offset = offset_from_grid(pos);
        let x_offset = grid_offset.x();
        let y_offset = grid_offset.y();
        if y_offset.abs() < x_offset.abs() && y_offset.abs() < 0.25 {
            Glyph::get_smooth_horizontal_glyphs_for_colored_floating_square(pos, color)
        } else if x_offset.abs() < 0.25 {
            Glyph::get_smooth_vertical_glyphs_for_colored_floating_square(pos, color)
        } else {
            Glyph::get_half_grid_glyphs_for_colored_floating_square(pos, color)
        }
    }
    pub fn get_smooth_horizontal_glyphs_for_floating_square(
        pos: Point<f32>,
    ) -> Vec<Vec<Option<Glyph>>> {
        Glyph::get_smooth_horizontal_glyphs_for_colored_floating_square(pos, ColorName::White)
    }
    pub fn get_smooth_horizontal_glyphs_for_colored_floating_square(
        pos: Point<f32>,
        color: ColorName,
    ) -> Vec<Vec<Option<Glyph>>> {
        let width = 3;
        let mut output = vec![vec![None; width]; width];

        let c = width / 2 as usize;

        let grid_offset = offset_from_grid(pos);
        let x_offset = grid_offset.x();
        let offset_dir = round(sign(grid_offset));

        for i in 0..3 {
            let x = i as i32 - 1;
            if offset_dir.x() == x || x == 0 {
                output[i][c] = Some(Glyph::colored_square_with_horizontal_offset(
                    x_offset - x as f32,
                    color,
                ));
            }
        }

        return output;
    }
    pub fn get_smooth_vertical_glyphs_for_floating_square(
        pos: Point<f32>,
    ) -> Vec<Vec<Option<Glyph>>> {
        Glyph::get_smooth_vertical_glyphs_for_colored_floating_square(pos, ColorName::White)
    }
    pub fn get_smooth_vertical_glyphs_for_colored_floating_square(
        pos: Point<f32>,
        color: ColorName,
    ) -> Vec<Vec<Option<Glyph>>> {
        let width = 3;
        let mut output = vec![vec![None; width]; width];

        let c = width / 2 as usize;

        let grid_offset = offset_from_grid(pos);
        let y_offset = grid_offset.y();
        let offset_dir = round(sign(grid_offset));
        for j in 0..3 {
            let y = j as i32 - 1;
            if offset_dir.y() == y || y == 0 {
                output[c][j] = Some(Glyph::colored_square_with_vertical_offset(
                    y_offset - y as f32,
                    color,
                ));
            }
        }
        return output;
    }

    pub fn get_half_grid_glyphs_for_floating_square(pos: Point<f32>) -> Vec<Vec<Option<Glyph>>> {
        Glyph::get_half_grid_glyphs_for_colored_floating_square(pos, ColorName::White)
    }

    pub fn get_half_grid_glyphs_for_colored_floating_square(
        pos: Point<f32>,
        color: ColorName,
    ) -> Vec<Vec<Option<Glyph>>> {
        let width = 3;
        let mut output = vec![vec![None; width]; width];
        let grid_offset = offset_from_grid(pos);
        let offset_dir = round(sign(grid_offset));

        for i in 0..3 {
            for j in 0..3 {
                let x = i as i32 - 1;
                let y = j as i32 - 1;
                let square = p(x as f32, y as f32);
                if (offset_dir.x() == x || x == 0) && (offset_dir.y() == y || y == 0) {
                    let glyph =
                        Glyph::colored_square_with_half_step_offset(grid_offset - square, color);
                    if glyph.character != ' ' {
                        output[i][j] = Some(glyph);
                    }
                }
            }
        }
        return output;
    }

    pub fn array_to_braille(input: BrailleArray) -> char {
        let mut dot_val: u32 = 0;
        for x in 0..2 {
            for y in 0..4 {
                if input[x][y] {
                    dot_val |= Glyph::braille_bit_for_pos(p(x as i32, y as i32));
                }
            }
        }
        return char::from_u32('\u{2800}' as u32 | dot_val).unwrap();
    }

    pub fn braille_bit_for_pos(p: Point<i32>) -> u32 {
        let braille_value_map = vec![vec![7, 3, 2, 1], vec![8, 6, 5, 4]];
        1 << (braille_value_map[p.x() as usize][p.y() as usize] - 1)
    }

    pub fn add_braille_dot(character: char, p: Point<i32>) -> char {
        char::from_u32(character as u32 | Glyph::braille_bit_for_pos(p)).unwrap()
    }

    pub fn empty_braille() -> char {
        '\u{2800}'
    }

    pub fn is_braille(c: char) -> bool {
        let x = c as u32;
        // The unicode braille block
        x >= 0x2800 && x <= 0x28FF
    }

    pub fn count_braille_dots(character: char) -> i32 {
        if !Glyph::is_braille(character) {
            return 0;
        }
        let num_good_bits = 8;
        let mut sum = 0;
        let bits = character as u32;
        for i in 0..num_good_bits {
            sum += (bits >> i) & 1;
        }
        return sum as i32;
    }

    pub fn add_braille(c1: char, c2: char) -> char {
        char::from_u32(c1 as u32 | c2 as u32).unwrap()
    }

    pub fn get_glyphs_for_braille_line(
        start_pos: Point<f32>,
        end_pos: Point<f32>,
    ) -> Vec<Vec<Option<Glyph>>> {
        Glyph::get_glyphs_for_colored_braille_line(start_pos, end_pos, ColorName::White)
    }

    pub fn world_pos_to_braille_pos(pos: Point<f32>) -> Point<f32> {
        p(pos.x() * 2.0 + 0.5, pos.y() * 4.0 + 1.5)
    }

    pub fn braille_pos_to_world_pos(pos: Point<f32>) -> Point<f32> {
        p((pos.x() - 0.5) / 2.0, (pos.y() - 1.5) / 4.0)
    }

    pub fn braille_square_to_dot_in_character(pos: Point<i32>) -> Point<i32> {
        p((pos.x() % 2).abs(), (pos.y() % 4).abs())
    }

    pub fn braille_grid_to_character_grid(braille_square: Point<i32>) -> Point<i32> {
        p(
            ((braille_square.x() as f32 - 0.5) / 2.0).round() as i32,
            ((braille_square.y() as f32 - 1.5) / 4.0).round() as i32,
        )
    }

    pub fn get_glyphs_for_colored_braille_line(
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        color: ColorName,
    ) -> Vec<Vec<Option<Glyph>>> {
        let start_grid_square = snap_to_grid(start_pos);
        let end_grid_square = snap_to_grid(end_pos);
        let start_braille_grid_square = snap_to_grid(Glyph::world_pos_to_braille_pos(start_pos));
        let end_braille_grid_square = snap_to_grid(Glyph::world_pos_to_braille_pos(end_pos));

        let grid_diagonal = end_grid_square - start_grid_square;
        let grid_width = grid_diagonal.x().abs() + 1;
        let grid_height = grid_diagonal.y().abs() + 1;

        let bottom_square_y = min(start_grid_square.y(), end_grid_square.y());
        let left_square_x = min(start_grid_square.x(), end_grid_square.x());
        let grid_origin_square = p(left_square_x, bottom_square_y);

        let mut output_grid: Vec<Vec<Option<Glyph>>> =
            vec![vec![None; grid_height as usize]; grid_width as usize];

        for (x, y) in line_drawing::Bresenham::new(
            start_braille_grid_square.x_y(),
            end_braille_grid_square.x_y(),
        ) {
            let character_grid_square =
                Glyph::braille_grid_to_character_grid(p(x, y)) - grid_origin_square;
            let glyph_in_grid = &mut output_grid[character_grid_square.x() as usize]
                [character_grid_square.y() as usize];
            if *glyph_in_grid == None {
                *glyph_in_grid = Some(Glyph {
                    character: Glyph::empty_braille(),
                    fg_color: color,
                    bg_color: ColorName::Black,
                });
            }
            let braille_character = &mut (*glyph_in_grid).as_mut().unwrap().character;
            *braille_character = Glyph::add_braille_dot(
                *braille_character,
                Glyph::braille_square_to_dot_in_character(p(x, y)),
            );
        }
        return output_grid;
    }

    pub fn world_pos_to_braille_char(world_pos: Point<f32>) -> char {
        let character = Glyph::empty_braille();
        Glyph::add_braille_dot(
            character,
            Glyph::braille_square_to_dot_in_character(snap_to_grid(
                Glyph::world_pos_to_braille_pos(world_pos),
            )),
        )
    }

    pub fn world_pos_to_colored_braille_glyph(world_pos: Point<f32>, color: ColorName) -> Glyph {
        Glyph {
            character: Glyph::world_pos_to_braille_char(world_pos),
            fg_color: color,
            bg_color: ColorName::Black,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;

    #[test]
    fn test_colored_square_with_half_step_offsets() {
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.0, 0.0), ColorName::Red).character
                == quarter_block_by_offset((0, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.0, 0.0), ColorName::Red).fg_color
                == ColorName::Red
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.0, 0.0), ColorName::Red).bg_color
                == ColorName::Black
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.1, 0.1), ColorName::Red).character
                == quarter_block_by_offset((0, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.24, 0.0), ColorName::Red).character
                == quarter_block_by_offset((0, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.25, 0.0), ColorName::Red).character
                == quarter_block_by_offset((1, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.26, 0.0), ColorName::Red).character
                == quarter_block_by_offset((1, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(-0.25, 0.0), ColorName::Red).character
                == quarter_block_by_offset((0, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(-0.26, 0.0), ColorName::Red).character
                == quarter_block_by_offset((-1, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.49, 0.0), ColorName::Red).character
                == quarter_block_by_offset((1, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.5, 0.0), ColorName::Red).character
                == quarter_block_by_offset((1, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.2, 0.4), ColorName::Red).character
                == quarter_block_by_offset((0, 1))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(-0.499, 0.4), ColorName::Red).character
                == quarter_block_by_offset((-1, 1))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.74, 0.0), ColorName::Red).character
                == quarter_block_by_offset((1, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.76, 0.0), ColorName::Red).character
                == quarter_block_by_offset((2, 0))
        );
        assert!(
            Glyph::colored_square_with_half_step_offset(p(0.3, -0.6), ColorName::Red).character
                == quarter_block_by_offset((1, -1))
        );
    }
    #[test]
    fn test_half_grid_glyph_when_rounding_to_zero_for_both_axes() {
        let test_pos = p(-0.24, 0.01);
        let glyphs = Glyph::get_half_grid_glyphs_for_floating_square(test_pos);
        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1] == None);
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((0, 0)));
        assert!(glyphs[1][2] == None);
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1] == None);
        assert!(glyphs[2][2] == None);
    }
    #[test]
    fn test_half_grid_glyphs_when_rounding_to_zero_for_x_and_half_step_up_for_y() {
        let test_pos = p(0.24, 0.26);
        let glyphs = Glyph::get_half_grid_glyphs_for_floating_square(test_pos);
        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1] == None);
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((0, 1)));
        assert!(glyphs[1][2].clone().unwrap().character == quarter_block_by_offset((0, -1)));
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1] == None);
        assert!(glyphs[2][2] == None);
    }
    #[test]
    fn test_half_grid_glyphs_when_rounding_to_zero_for_x_and_exactly_half_step_up_for_y() {
        let test_pos = p(0.24, 0.25);

        let glyphs = Glyph::get_half_grid_glyphs_for_floating_square(test_pos);
        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1] == None);
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((0, 1)));
        assert!(glyphs[1][2].clone().unwrap().character == quarter_block_by_offset((0, -1)));
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1] == None);
        assert!(glyphs[2][2] == None);
    }
    #[test]
    fn test_half_grid_glyphs_when_rounding_to_zero_for_x_and_exactly_half_step_down_for_y() {
        let test_pos = p(-0.2, -0.25);
        let glyphs = Glyph::get_half_grid_glyphs_for_floating_square(test_pos);
        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1] == None);
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((0, 0)));
        assert!(glyphs[1][2] == None);
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1] == None);
        assert!(glyphs[2][2] == None);
    }
    #[test]
    fn test_half_grid_glyphs_when_rounding_to_zero_for_y_and_half_step_right_for_x() {
        let test_pos = p(0.3, 0.1);
        let glyphs = Glyph::get_half_grid_glyphs_for_floating_square(test_pos);
        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1] == None);
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((1, 0)));
        assert!(glyphs[1][2] == None);
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1].clone().unwrap().character == quarter_block_by_offset((-1, 0)));
        assert!(glyphs[2][2] == None);
    }

    #[test]
    fn test_half_grid_glyphs_when_rounding_to_zero_for_y_and_half_step_left_for_x() {
        let test_pos = p(-0.3, 0.2);
        let glyphs = Glyph::get_half_grid_glyphs_for_floating_square(test_pos);
        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1].clone().unwrap().character == quarter_block_by_offset((1, 0)));
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((-1, 0)));
        assert!(glyphs[1][2] == None);
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1] == None);
        assert!(glyphs[2][2] == None);
    }
    #[test]
    fn test_array_to_braille_char() {
        // 10
        // 00
        // 01
        // 01
        let input = [[false, false, false, true], [true, true, false, false]];
        assert!(Glyph::array_to_braille(input) == '⢡');

        // 00
        // 11
        // 01
        // 00
        let input = [[false, false, true, false], [false, true, true, false]];
        assert!(Glyph::array_to_braille(input) == '⠲');
    }

    #[test]
    fn test_get_empty_braille_character() {
        assert!(Glyph::empty_braille() == '\u{2800}');
    }

    #[test]
    fn test_set_braille_dot() {
        let mut b = Glyph::empty_braille();
        b = Glyph::add_braille_dot(b, p(0, 0));
        b = Glyph::add_braille_dot(b, p(1, 1));
        assert!(b == '⡠');
    }

    #[test]
    fn test_chars_for_horizontal_braille_line_without_rounding() {
        let start = p(-0.25, -0.4);
        let end = p(1.75, -0.4);

        // Expected braille:
        // 00 00 00
        // 00 00 00
        // 00 00 00
        // 11 11 10

        let line_glyphs = Glyph::get_glyphs_for_braille_line(start, end);
        assert!(line_glyphs.len() == 3);
        assert!(line_glyphs[0].len() == 1);
        assert!(line_glyphs[1].len() == 1);
        assert!(line_glyphs[2].len() == 1);

        assert!(line_glyphs[0][0].clone().unwrap().character == '\u{28C0}');
        assert!(line_glyphs[1][0].clone().unwrap().character == '\u{28C0}');
        assert!(line_glyphs[2][0].clone().unwrap().character == '\u{2840}');
    }

    #[test]
    fn test_chars_for_horizontal_braille_line_with_offset_without_rounding() {
        let start = p(-0.25, 0.4);
        let end = p(1.75, 0.4);

        // Expected braille:
        // 11 11 10
        // 00 00 00
        // 00 00 00
        // 00 00 00

        let line_glyphs = Glyph::get_glyphs_for_braille_line(start, end);
        assert!(line_glyphs.len() == 3);
        assert!(line_glyphs[0].len() == 1);
        assert!(line_glyphs[1].len() == 1);
        assert!(line_glyphs[2].len() == 1);

        assert!(line_glyphs[0][0].clone().unwrap().character == '\u{2809}');
        assert!(line_glyphs[1][0].clone().unwrap().character == '\u{2809}');
        assert!(line_glyphs[2][0].clone().unwrap().character == '\u{2801}');
    }

    #[test]
    fn test_chars_for_vertical_braille_line_without_rounding() {
        let start = p(-0.25, -0.4);
        let end = p(-0.25, 0.875);

        // Expected braille:
        // 00
        // 00
        // 10
        // 10

        // 10
        // 10
        // 10
        // 10

        let line_glyphs = Glyph::get_glyphs_for_braille_line(start, end);
        assert!(line_glyphs.len() == 1);
        assert!(line_glyphs[0].len() == 2);

        assert!(line_glyphs[0][0].clone().unwrap().character == '\u{2847}');
        assert!(line_glyphs[0][1].clone().unwrap().character == '\u{2844}');
    }

    #[test]
    fn test_braille_grid_to_character_grid() {
        assert!(Glyph::braille_grid_to_character_grid(p(0, 0)) == p(0, 0));
        assert!(Glyph::braille_grid_to_character_grid(p(1, 3)) == p(0, 0));
        assert!(Glyph::braille_grid_to_character_grid(p(-1, -1)) == p(-1, -1));
        assert!(Glyph::braille_grid_to_character_grid(p(2, 8)) == p(1, 2));
        assert!(Glyph::braille_grid_to_character_grid(p(21, 80)) == p(10, 20));
    }

    #[test]
    fn test_world_pos_to_braille_pos() {
        assert!(Glyph::world_pos_to_braille_pos(p(0.0, 0.0)) == p(0.5, 1.5));
        assert!(Glyph::world_pos_to_braille_pos(p(1.0, 0.0)) == p(2.5, 1.5));
        assert!(Glyph::world_pos_to_braille_pos(p(0.25, 0.375)) == p(1.0, 3.0));
    }

    #[test]
    fn test_braille_pos_to_world_pos() {
        assert!(Glyph::braille_pos_to_world_pos(p(0.5, 1.5)) == p(0.0, 0.0));
        assert!(Glyph::braille_pos_to_world_pos(p(2.5, 1.5)) == p(1.0, 0.0));
        assert!(Glyph::braille_pos_to_world_pos(p(1.0, 3.0)) == p(0.25, 0.375));
    }

    #[test]
    fn test_braille_square_to_dot_in_character() {
        assert!(Glyph::braille_square_to_dot_in_character(p(0, 0)) == p(0, 0));
        assert!(Glyph::braille_square_to_dot_in_character(p(1, 3)) == p(1, 3));
        assert!(Glyph::braille_square_to_dot_in_character(p(25, 4)) == p(1, 0));
        assert!(Glyph::braille_square_to_dot_in_character(p(-3, 4)) == p(1, 0));
    }

    #[test]
    fn test_combine_braille_character() {
        assert!(Glyph::add_braille('\u{2800}', '\u{2820}') == '\u{2820}');
        assert!(Glyph::add_braille('\u{2801}', '\u{28C0}') == '\u{28C1}');
    }

    #[test]
    fn test_world_point_to_braille_char() {
        assert!(Glyph::world_pos_to_braille_char(p(0.0, 0.0)) == '\u{2810}');
        assert!(Glyph::world_pos_to_braille_char(p(-0.4, -0.4)) == '\u{2840}');
        assert!(Glyph::world_pos_to_braille_char(p(0.2, 0.4)) == '\u{2808}');
    }

    #[test]
    fn test_world_point_to_braille_glyph() {
        let points = [p(0.0, 0.0), p(-0.4, -0.4), p(0.2, 0.4)];
        for p1 in points {
            assert!(
                Glyph::world_pos_to_colored_braille_glyph(p1, ColorName::Black).character
                    == Glyph::world_pos_to_braille_char(p1)
            );
        }
    }

    #[test]
    fn test_count_braille_dots() {
        assert!(Glyph::count_braille_dots('\u{2800}') == 0);
        assert!(Glyph::count_braille_dots('\u{2818}') == 2);
        assert!(Glyph::count_braille_dots('\u{28C0}') == 2);
        assert!(Glyph::count_braille_dots('\u{28FF}') == 8);
        assert!(Glyph::count_braille_dots('A') == 0);
        assert!(Glyph::count_braille_dots('#') == 0);
    }
}
