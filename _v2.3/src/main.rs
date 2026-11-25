#![windows_subsystem = "windows"]

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc::Receiver, Arc};
use std::time::{Duration, Instant};

use eframe::egui;
use las::{Reader, Writer};
use rayon::prelude::*;
use rfd::FileDialog;
use roaring::RoaringBitmap;
use seahash::SeaHasher;
use uuid::Uuid;

// Поддерживаемые форматы облаков точек
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PointCloudFormat {
    Las,
    E57,
}

// Способы прореживания
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinningMethod {
    /// По коэффициенту сетки k x k (с сохранением структуры и фотопанорам от 1-го лица)
    GridStride,
    /// По среднему расстоянию (3D-сетка вокселей, универсально)
    VoxelGrid,
}

// Ограничивающий параллелепипед для обрезки (Limit Box)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LimitBox {
    pub enabled: bool,
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
    pub min_z: f64,
    pub max_z: f64,
    pub yaw_deg: f64,
}

impl Default for LimitBox {
    fn default() -> Self {
        Self {
            enabled: false,
            min_x: 0.0,
            max_x: 0.0,
            min_y: 0.0,
            max_y: 0.0,
            min_z: 0.0,
            max_z: 0.0,
            yaw_deg: 0.0,
        }
    }
}

impl LimitBox {
    #[inline]
    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        if !self.enabled {
            return true;
        }
        if z < self.min_z || z > self.max_z {
            return false;
        }
        if self.yaw_deg.abs() < 1e-6 {
            x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
        } else {
            let cx = (self.min_x + self.max_x) * 0.5;
            let cy = (self.min_y + self.max_y) * 0.5;
            let dx = x - cx;
            let dy = y - cy;
            let (sin_a, cos_a) = (-self.yaw_deg.to_radians()).sin_cos();
            let lx = cx + dx * cos_a - dy * sin_a;
            let ly = cy + dx * sin_a + dy * cos_a;
            lx >= self.min_x && lx <= self.max_x && ly >= self.min_y && ly <= self.max_y
        }
    }

    pub fn corners(&self) -> [(f64, f64, f64); 8] {
        let raw = [
            (self.min_x, self.min_y, self.min_z),
            (self.max_x, self.min_y, self.min_z),
            (self.max_x, self.max_y, self.min_z),
            (self.min_x, self.max_y, self.min_z),
            (self.min_x, self.min_y, self.max_z),
            (self.max_x, self.min_y, self.max_z),
            (self.max_x, self.max_y, self.max_z),
            (self.min_x, self.max_y, self.max_z),
        ];
        if self.yaw_deg.abs() < 1e-6 {
            return raw;
        }
        let cx = (self.min_x + self.max_x) * 0.5;
        let cy = (self.min_y + self.max_y) * 0.5;
        let (sin_a, cos_a) = self.yaw_deg.to_radians().sin_cos();
        let mut out = [(0.0, 0.0, 0.0); 8];
        for (i, &(x, y, z)) in raw.iter().enumerate() {
            let dx = x - cx;
            let dy = y - cy;
            out[i] = (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a, z);
        }
        out
    }

    pub fn dimensions(&self) -> (f64, f64, f64) {
        (
            (self.max_x - self.min_x).abs(),
            (self.max_y - self.min_y).abs(),
            (self.max_z - self.min_z).abs(),
        )
    }

    pub fn volume(&self) -> f64 {
        let (dx, dy, dz) = self.dimensions();
        dx * dy * dz
    }

    pub fn reset_to_bounds(&mut self, min: (f64, f64, f64), max: (f64, f64, f64)) {
        self.min_x = min.0;
        self.max_x = max.0;
        self.min_y = min.1;
        self.max_y = max.1;
        self.min_z = min.2;
        self.max_z = max.2;
        self.yaw_deg = 0.0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    Rgb,
    Height,
}

// Точка для быстрого 3D предпросмотра
#[derive(Clone, Debug)]
pub struct PreviewPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub color: Option<egui::Color32>,
}

// Состояние 3D-окна просмотра
#[derive(Clone, Debug)]
pub struct Viewport3D {
    pub show_window: bool,
    pub canvas_height: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub pan_offset: egui::Vec2,
    pub point_size: f32,
    pub hide_outside: bool,
    pub color_mode: ColorMode,
    pub cloud_center: (f64, f64, f64),
    pub cloud_extent: (f64, f64, f64),
    pub bounds_min: (f64, f64, f64),
    pub bounds_max: (f64, f64, f64),
}

impl Default for Viewport3D {
    fn default() -> Self {
        Self {
            show_window: false,
            canvas_height: 580.0,
            yaw: -0.785,   // -45 градусов
            pitch: 0.52,   // ~30 градусов (вид сверху-сбоку)
            distance: 100.0,
            pan_offset: egui::Vec2::ZERO,
            point_size: 2.0,
            hide_outside: false,
            color_mode: ColorMode::Rgb,
            cloud_center: (0.0, 0.0, 0.0),
            cloud_extent: (100.0, 100.0, 50.0),
            bounds_min: (0.0, 0.0, 0.0),
            bounds_max: (100.0, 100.0, 50.0),
        }
    }
}

// Проекция 3D точки на плоскость экрана (LookAt орбитальная камера вокруг центра)
pub fn project_3d_point(
    p: (f64, f64, f64),
    center: (f64, f64, f64),
    yaw: f32,
    pitch: f32,
    distance: f32,
    screen_center: egui::Pos2,
    focal: f32,
) -> Option<(egui::Pos2, f32)> {
    let rx = (p.0 - center.0) as f32;
    let ry = (p.1 - center.1) as f32;
    let rz = (p.2 - center.2) as f32;

    let (sin_y, cos_y) = yaw.sin_cos();
    let (sin_p, cos_p) = pitch.sin_cos();

    // LookAt Orbit камера вокруг центра (Z - вверх, Y - север/вглубь, X - восток/вправо)
    // Вектор вправо камеры: (cos_y, sin_y, 0)
    let x_cam = rx * cos_y + ry * sin_y;

    // Вектор вверх камеры: (-sin_y * sin_p, cos_y * sin_p, cos_p)
    let y_cam = -rx * sin_y * sin_p + ry * cos_y * sin_p + rz * cos_p;

    // Глубина z_depth вдоль луча зрения:
    let depth = distance - rx * sin_y * cos_p + ry * cos_y * cos_p - rz * sin_p;
    if depth <= 0.5 {
        return None;
    }

    let scale = focal / depth;
    let sx = screen_center.x + x_cam * scale;
    let sy = screen_center.y - y_cam * scale;

    Some((egui::pos2(sx, sy), depth))
}

// Градиент высоты (Rainbow / Turbo): Синий -> Бирюзовый -> Зеленый -> Желтый -> Красный
pub fn height_to_color(z: f64, min_z: f64, max_z: f64, is_inside: bool) -> egui::Color32 {
    if !is_inside {
        return egui::Color32::from_rgba_unmultiplied(70, 70, 80, 45);
    }
    let range = (max_z - min_z).max(0.001);
    let t = ((z - min_z) / range).clamp(0.0, 1.0);

    if t < 0.25 {
        let f = (t / 0.25) as f32;
        egui::Color32::from_rgb(0, (f * 255.0) as u8, 255)
    } else if t < 0.5 {
        let f = ((t - 0.25) / 0.25) as f32;
        egui::Color32::from_rgb(0, 255, ((1.0 - f) * 255.0) as u8)
    } else if t < 0.75 {
        let f = ((t - 0.5) / 0.25) as f32;
        egui::Color32::from_rgb((f * 255.0) as u8, 255, 0)
    } else {
        let f = ((t - 0.75) / 0.25) as f32;
        egui::Color32::from_rgb(255, ((1.0 - f) * 255.0) as u8, 0)
    }
}

// Хэширование координат вокселя в u64
fn hash_voxel<T: Hash>(t: &T) -> u64 {
    let mut s = SeaHasher::with_seeds(0, 0, 0, 0);
    t.hash(&mut s);
    s.finish()
}

// Поворот и перенос точки матрицей положения скана (Transform / Pose)
fn transform_point(point: (f64, f64, f64), transform: &Option<e57::Transform>) -> (f64, f64, f64) {
    if let Some(t) = transform {
        let (px, py, pz) = point;
        let q = &t.rotation;

        // t_vec = 2 * (q_xyz x p)
        let tx = 2.0 * (q.y * pz - q.z * py);
        let ty = 2.0 * (q.z * px - q.x * pz);
        let tz = 2.0 * (q.x * py - q.y * px);

        // v' = p + q.w * t_vec + (q_xyz x t_vec)
        let rx = px + q.w * tx + (q.y * tz - q.z * ty);
        let ry = py + q.w * ty + (q.z * tx - q.x * tz);
        let rz = pz + q.w * tz + (q.x * ty - q.y * tx);

        (
            rx + t.translation.x,
            ry + t.translation.y,
            rz + t.translation.z,
        )
    } else {
        point
    }
}

// Определение формата по расширению
fn detect_format(path: &Path) -> Option<PointCloudFormat> {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        let ext_lower = ext.to_lowercase();
        if ext_lower == "las" || ext_lower == "laz" {
            return Some(PointCloudFormat::Las);
        } else if ext_lower == "e57" {
            return Some(PointCloudFormat::E57);
        }
    }
    None
}

// Добавление расширения при сохранении, если оно отсутствует
fn add_extension_if_needed(path: &Path, default_ext: &str) -> PathBuf {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        let ext_lower = ext.to_lowercase();
        if ext_lower == "las" || ext_lower == "laz" || ext_lower == "e57" {
            return path.to_path_buf();
        }
    }
    let mut p = path.to_path_buf();
    p.set_extension(default_ext);
    p
}

// Форматирование времени
fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let secs = secs % 60;

    if hours > 0 {
        format!("{} ч {} мин {} сек", hours, mins, secs)
    } else if mins > 0 {
        format!("{} мин {} сек", mins, secs)
    } else {
        format!("{} сек", secs)
    }
}

// Форматирование чисел с разделителем тысяч пробелом (например, 163 375 183)
fn format_number_with_spaces(mut n: usize) -> String {
    if n == 0 {
        return "0".to_string();
    }
    let mut s = String::new();
    let mut count = 0;
    while n > 0 {
        if count > 0 && count % 3 == 0 {
            s.push(' ');
        }
        s.push(char::from_digit((n % 10) as u32, 10).unwrap());
        n /= 10;
        count += 1;
    }
    s.chars().rev().collect()
}

// Форматирование размера файла (байты, КБ, МБ, ГБ)
fn format_file_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} ГБ", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} МБ", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{} КБ", bytes / 1024)
    } else {
        format!("{} байт", bytes)
    }
}

// Сообщения потоковой передачи точек предпросмотра
#[derive(Debug)]
pub enum PreviewStreamMsg {
    Chunk(Vec<PreviewPoint>),
    Finished,
    Error(String),
}

// Быстрый постраничный ридер E57 для мгновенного пропуска пакетов при 3D-предпросмотре
struct PagedFastReader {
    file: std::fs::File,
    page_size: u64,
    data_per_page: u64,
    log_file_size: u64,
    buf_start_page: u64,
    buf_pages_loaded: u64,
    buf_pages_cap: u64,
    buf: Vec<u8>,
}

impl PagedFastReader {
    fn open(path: &Path, page_size: u64, buf_pages_cap: u64) -> std::io::Result<Self> {
        use std::io::Seek;
        let mut file = std::fs::File::open(path)?;
        let phy_size = file.seek(std::io::SeekFrom::End(0))?;
        let data_per_page = page_size.saturating_sub(4).max(1);
        let pages = phy_size / page_size;
        let cap = buf_pages_cap.max(1);
        Ok(Self {
            file,
            page_size,
            data_per_page,
            log_file_size: pages * data_per_page,
            buf_start_page: 0,
            buf_pages_loaded: 0,
            buf_pages_cap: cap,
            buf: vec![0u8; (page_size * cap) as usize],
        })
    }

    #[inline]
    fn phys_to_log(&self, phys: u64) -> u64 {
        let pages_before = phys / self.page_size;
        phys.saturating_sub(pages_before * 4)
    }

    fn read_at_log(&mut self, log_off: u64, out: &mut [u8]) -> std::io::Result<()> {
        use std::io::{Read, Seek};
        let total_pages = self.log_file_size / self.data_per_page;
        let mut cur_log = log_off;
        let mut filled = 0;
        while filled < out.len() {
            if cur_log >= self.log_file_size {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "EOF"));
            }
            let page = cur_log / self.data_per_page;
            if page < self.buf_start_page || page >= self.buf_start_page + self.buf_pages_loaded {
                let pages_avail = total_pages.saturating_sub(page).min(self.buf_pages_cap);
                let byte_len = (pages_avail * self.page_size) as usize;
                self.file.seek(std::io::SeekFrom::Start(page * self.page_size))?;
                self.file.read_exact(&mut self.buf[..byte_len])?;
                self.buf_start_page = page;
                self.buf_pages_loaded = pages_avail;
            }
            let page_idx = (page - self.buf_start_page) as usize;
            let page_base = page_idx * self.page_size as usize;
            let in_page = (cur_log % self.data_per_page) as usize;
            let avail = (self.data_per_page as usize) - in_page;
            let to_read = (out.len() - filled).min(avail);
            out[filled..filled + to_read].copy_from_slice(&self.buf[page_base + in_page..page_base + in_page + to_read]);
            filled += to_read;
            cur_log += to_read as u64;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum FastAttr {
    Single,
    Double,
    ScaledInt { bytes: usize, min: i64, scale: f64, offset: f64 },
    Int { bytes: usize, min: i64 },
}

impl FastAttr {
    fn from_data_type(dt: &e57::RecordDataType) -> Option<Self> {
        match dt {
            e57::RecordDataType::Single { .. } => Some(Self::Single),
            e57::RecordDataType::Double { .. } => Some(Self::Double),
            e57::RecordDataType::ScaledInteger { min, max, scale, offset } => {
                let range = *max as i128 - *min as i128;
                let bits = if range <= 0 { 0 } else { range.ilog2() as usize + 1 };
                if bits > 0 && bits % 8 == 0 {
                    Some(Self::ScaledInt { bytes: bits / 8, min: *min, scale: *scale, offset: *offset })
                } else {
                    None
                }
            }
            e57::RecordDataType::Integer { min, max } => {
                let range = *max as i128 - *min as i128;
                let bits = if range <= 0 { 0 } else { range.ilog2() as usize + 1 };
                if bits > 0 && bits % 8 == 0 {
                    Some(Self::Int { bytes: bits / 8, min: *min })
                } else {
                    None
                }
            }
        }
    }

    #[inline]
    fn byte_size(self) -> usize {
        match self {
            Self::Single => 4,
            Self::Double => 8,
            Self::ScaledInt { bytes, .. } | Self::Int { bytes, .. } => bytes,
        }
    }

    #[inline]
    fn decode_f64(self, slice: &[u8], idx: usize) -> f64 {
        match self {
            Self::Single => {
                let off = idx * 4;
                f32::from_le_bytes([slice[off], slice[off + 1], slice[off + 2], slice[off + 3]]) as f64
            }
            Self::Double => {
                let off = idx * 8;
                f64::from_le_bytes([
                    slice[off], slice[off + 1], slice[off + 2], slice[off + 3],
                    slice[off + 4], slice[off + 5], slice[off + 6], slice[off + 7],
                ])
            }
            Self::ScaledInt { bytes, min, scale, offset } => {
                let off = idx * bytes;
                let mut arr = [0u8; 8];
                arr[..bytes].copy_from_slice(&slice[off..off + bytes]);
                let raw = u64::from_le_bytes(arr);
                ((raw as i64).wrapping_add(min)) as f64 * scale + offset
            }
            Self::Int { bytes, min } => {
                let off = idx * bytes;
                let mut arr = [0u8; 8];
                arr[..bytes].copy_from_slice(&slice[off..off + bytes]);
                let raw = u64::from_le_bytes(arr);
                ((raw as i64).wrapping_add(min)) as f64
            }
        }
    }
}

fn bit_size_of_record(dt: &e57::RecordDataType) -> usize {
    match dt {
        e57::RecordDataType::Single { .. } => 32,
        e57::RecordDataType::Double { .. } => 64,
        e57::RecordDataType::ScaledInteger { min, max, .. }
        | e57::RecordDataType::Integer { min, max } => {
            let range = *max as i128 - *min as i128;
            if range <= 0 { 0 } else { range.ilog2() as usize + 1 }
        }
    }
}

fn color_range_of(dt: &e57::RecordDataType, lim_min: &Option<e57::RecordValue>, lim_max: &Option<e57::RecordValue>) -> (f64, f64) {
    let dt_range = match dt {
        e57::RecordDataType::Single { min, max } => (min.unwrap_or(0.0) as f64, max.unwrap_or(1.0) as f64),
        e57::RecordDataType::Double { min, max } => (min.unwrap_or(0.0), max.unwrap_or(1.0)),
        e57::RecordDataType::ScaledInteger { min, max, scale, offset } => (
            *min as f64 * *scale + *offset,
            *max as f64 * *scale + *offset,
        ),
        e57::RecordDataType::Integer { min, max } => (*min as f64, *max as f64),
    };
    let lo = lim_min.as_ref().and_then(|v| v.to_f64(dt).ok()).unwrap_or(dt_range.0);
    let hi = lim_max.as_ref().and_then(|v| v.to_f64(dt).ok()).unwrap_or(dt_range.1);
    (lo, hi)
}

// Быстрая выборка точек скана E57 с перешагиванием пакетов по 360° (без полной распаковки)
// Возвращает Ok(Some(total_added)) если быстрый режим применим, или Ok(None) для безопасного fallback.
fn fast_sample_e57_scan(
    paged: &mut PagedFastReader,
    pc: &e57::PointCloud,
    target_pts: usize,
    phase_seed: usize,
    cancel: &AtomicBool,
    tx: &std::sync::mpsc::Sender<PreviewStreamMsg>,
) -> anyhow::Result<Option<usize>> {
    if pc.records == 0 || target_pts == 0 {
        return Ok(Some(0));
    }

    let find_rec = |name: e57::RecordName| pc.prototype.iter().position(|r| r.name == name);
    let (Some(cx_i), Some(cy_i), Some(cz_i)) = (
        find_rec(e57::RecordName::CartesianX),
        find_rec(e57::RecordName::CartesianY),
        find_rec(e57::RecordName::CartesianZ),
    ) else {
        return Ok(None);
    };

    let (Some(fx), Some(fy), Some(fz)) = (
        FastAttr::from_data_type(&pc.prototype[cx_i].data_type),
        FastAttr::from_data_type(&pc.prototype[cy_i].data_type),
        FastAttr::from_data_type(&pc.prototype[cz_i].data_type),
    ) else {
        return Ok(None);
    };

    let color_spec = match (
        find_rec(e57::RecordName::ColorRed),
        find_rec(e57::RecordName::ColorGreen),
        find_rec(e57::RecordName::ColorBlue),
    ) {
        (Some(ri), Some(gi), Some(bi)) => {
            match (
                FastAttr::from_data_type(&pc.prototype[ri].data_type),
                FastAttr::from_data_type(&pc.prototype[gi].data_type),
                FastAttr::from_data_type(&pc.prototype[bi].data_type),
            ) {
                (Some(fr), Some(fg), Some(fb)) => {
                    let cl = pc.color_limits.as_ref();
                    let rr = color_range_of(&pc.prototype[ri].data_type, &cl.and_then(|c| c.red_min.clone()), &cl.and_then(|c| c.red_max.clone()));
                    let gr = color_range_of(&pc.prototype[gi].data_type, &cl.and_then(|c| c.green_min.clone()), &cl.and_then(|c| c.green_max.clone()));
                    let br = color_range_of(&pc.prototype[bi].data_type, &cl.and_then(|c| c.blue_min.clone()), &cl.and_then(|c| c.blue_max.clone()));
                    Some((ri, gi, bi, fr, fg, fb, rr, gr, br))
                }
                _ => None,
            }
        }
        _ => None,
    };

    let inv_spec = find_rec(e57::RecordName::CartesianInvalidState).map(|idx| {
        (idx, bit_size_of_record(&pc.prototype[idx].data_type))
    });
    if let Some((_, bits)) = inv_spec {
        if bits == 0 || bits > 8 {
            return Ok(None);
        }
    }
    let col_inv_spec = find_rec(e57::RecordName::IsColorInvalid).map(|idx| {
        (idx, bit_size_of_record(&pc.prototype[idx].data_type))
    });
    if let Some((_, bits)) = col_inv_spec {
        if bits == 0 || bits > 8 {
            return Ok(None);
        }
    }

    let mut cv_hdr = [0u8; 32];
    let sec_start_log = paged.phys_to_log(pc.file_offset);
    if paged.read_at_log(sec_start_log, &mut cv_hdr).is_err() || cv_hdr[0] != 1 {
        return Ok(None);
    }
    let section_len = u64::from_le_bytes(cv_hdr[8..16].try_into().unwrap());
    let data_offset_phys = u64::from_le_bytes(cv_hdr[16..24].try_into().unwrap());
    let start_log = paged.phys_to_log(data_offset_phys);
    let end_log = (sec_start_log + section_len).min(paged.log_file_size);
    if end_log <= start_log {
        return Ok(Some(0));
    }

    let data_span = end_log - start_log;
    let mut first_p = [0u8; 6];
    let first_pkt_len = if paged.read_at_log(start_log, &mut first_p).is_ok() && (first_p[0] == 0 || first_p[0] == 1 || first_p[0] == 2) {
        (u16::from_le_bytes([first_p[2], first_p[3]]) as u64 + 1).max(64)
    } else {
        49_152
    };
    let est_total_pkts = (data_span / first_pkt_len).max(1) as usize;
    let pkts_wanted = if target_pts >= pc.records as usize {
        est_total_pkts
    } else {
        (target_pts / 25).clamp(1, est_total_pkts)
    };
    let milestone_step = (data_span / pkts_wanted as u64).max(1);
    let pts_per_pkt = target_pts.div_ceil(pkts_wanted).max(1);

    let proto_len = pc.prototype.len();
    let mut cur = start_log;
    let mut next_milestone = start_log + if phase_seed == 0 {
        0
    } else {
        (milestone_step * ((phase_seed as u64 * 37) % 100)) / 100
    };
    let mut deficit_pts: usize = 0;
    let mut hdr_buf = vec![0u8; 6 + proto_len * 2];
    let mut payload_buf = Vec::with_capacity(65536);
    let mut bs_sizes = vec![0usize; proto_len];
    let mut stream_offs = vec![0usize; proto_len];

    let mut cum_pts: u64 = 0;
    let mut cum_inv_bytes: u64 = 0;
    let mut cum_col_inv_bytes: u64 = 0;

    let mut out_chunk = Vec::with_capacity(2048);
    let mut total_added = 0usize;
    let mut last_send = Instant::now();
    let x_bytes = fx.byte_size();

    while cur + 6 <= end_log && total_added < target_pts {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let mut p4 = [0u8; 6];
        if paged.read_at_log(cur, &mut p4).is_err() {
            break;
        }
        let pid = p4[0];
        let plen_pk = u16::from_le_bytes([p4[2], p4[3]]) as u64 + 1;
        if pid == 0 || pid == 2 {
            cur += plen_pk;
            if cur % 4 != 0 {
                cur += 4 - (cur % 4);
            }
            continue;
        }
        if pid != 1 {
            return Ok(None);
        }
        let bsc = u16::from_le_bytes([p4[4], p4[5]]) as usize;
        if bsc != proto_len {
            return Ok(None);
        }
        if hdr_buf.len() < 6 + bsc * 2 {
            hdr_buf.resize(6 + bsc * 2, 0);
        }
        if paged.read_at_log(cur + 6, &mut hdr_buf[6..6 + bsc * 2]).is_err() {
            break;
        }
        let mut total_bs = 0usize;
        for i in 0..bsc {
            let sz = u16::from_le_bytes([hdr_buf[6 + i * 2], hdr_buf[6 + i * 2 + 1]]) as usize;
            stream_offs[i] = total_bs;
            bs_sizes[i] = sz;
            total_bs += sz;
        }

        let pkt_pts = bs_sizes[cx_i] / x_bytes;
        let pts_before = cum_pts;
        cum_pts += pkt_pts as u64;

        let mut inv_aligned = true;
        if let Some((inv_i, inv_bits)) = inv_spec {
            let lo_before = (cum_inv_bytes * 8) as i64 - (pts_before * inv_bits as u64) as i64;
            cum_inv_bytes += bs_sizes[inv_i] as u64;
            let lo_after = (cum_inv_bytes * 8) as i64 - (cum_pts * inv_bits as u64) as i64;
            inv_aligned = lo_before == 0 && lo_after >= 0;
        }
        let mut col_inv_aligned = true;
        if let Some((ci_i, ci_bits)) = col_inv_spec {
            let lo_before = (cum_col_inv_bytes * 8) as i64 - (pts_before * ci_bits as u64) as i64;
            cum_col_inv_bytes += bs_sizes[ci_i] as u64;
            let lo_after = (cum_col_inv_bytes * 8) as i64 - (cum_pts * ci_bits as u64) as i64;
            col_inv_aligned = lo_before == 0 && lo_after >= 0;
        }

        if cur >= next_milestone {
            deficit_pts = deficit_pts.saturating_add(pts_per_pkt);
            while next_milestone <= cur {
                next_milestone += milestone_step;
            }
        }

        if deficit_pts > 0 && pkt_pts > 0 && inv_aligned && col_inv_aligned && total_bs > 0 {
            payload_buf.resize(total_bs, 0);
            if paged.read_at_log(cur + 6 + (bsc * 2) as u64, &mut payload_buf).is_ok() {
                let x_slice = &payload_buf[stream_offs[cx_i]..stream_offs[cx_i] + bs_sizes[cx_i]];
                let y_slice = &payload_buf[stream_offs[cy_i]..stream_offs[cy_i] + bs_sizes[cy_i]];
                let z_slice = &payload_buf[stream_offs[cz_i]..stream_offs[cz_i] + bs_sizes[cz_i]];

                // Сначала отбираем валидные индексы пакета (чтобы пустое небо не съедало квоту выборки)
                let mut valid_indices = Vec::with_capacity(pkt_pts);
                if let Some((inv_i, inv_bits)) = inv_spec {
                    let inv_slice = &payload_buf[stream_offs[inv_i]..stream_offs[inv_i] + bs_sizes[inv_i]];
                    let mask = (1u16 << inv_bits) - 1;
                    for idx_p in 0..pkt_pts {
                        let bit_pos = idx_p * inv_bits;
                        let byte_pos = bit_pos / 8;
                        let bit_off = bit_pos % 8;
                        let raw_u16 = if byte_pos + 1 < inv_slice.len() {
                            u16::from_le_bytes([inv_slice[byte_pos], inv_slice[byte_pos + 1]])
                        } else {
                            inv_slice[byte_pos] as u16
                        };
                        if ((raw_u16 >> bit_off) & mask) == 0 {
                            valid_indices.push(idx_p);
                        }
                    }
                } else {
                    valid_indices.extend(0..pkt_pts);
                }

                if !valid_indices.is_empty() {
                    let want_now = deficit_pts.min(target_pts - total_added);
                    let step_in_pkt = if valid_indices.len() <= want_now {
                        1
                    } else {
                        (valid_indices.len() / want_now).max(1)
                    };
                    let mut got = 0usize;
                    let pkt_phase = if phase_seed == 0 { 0 } else { (phase_seed * 7) % step_in_pkt };
                    for &j in valid_indices.iter().skip(pkt_phase).step_by(step_in_pkt) {
                        if got >= want_now || total_added >= target_pts {
                            break;
                        }
                        let lx = fx.decode_f64(x_slice, j);
                        let ly = fy.decode_f64(y_slice, j);
                        let lz = fz.decode_f64(z_slice, j);
                        let (wx, wy, wz) = transform_point((lx, ly, lz), &pc.transform);

                        let color = if let Some((ri, gi, bi, fr, fg, fb, rr, gr, br)) = color_spec {
                            let col_valid = if let Some((ci_i, ci_bits)) = col_inv_spec {
                                let ci_slice = &payload_buf[stream_offs[ci_i]..stream_offs[ci_i] + bs_sizes[ci_i]];
                                let bit_pos = j * ci_bits;
                                let byte_pos = bit_pos / 8;
                                let bit_off = bit_pos % 8;
                                let mask = (1u8 << ci_bits) - 1;
                                ((ci_slice[byte_pos] >> bit_off) & mask) == 0
                            } else {
                                true
                            };
                            if col_valid {
                                let r_v = fr.decode_f64(&payload_buf[stream_offs[ri]..stream_offs[ri] + bs_sizes[ri]], j);
                                let g_v = fg.decode_f64(&payload_buf[stream_offs[gi]..stream_offs[gi] + bs_sizes[gi]], j);
                                let b_v = fb.decode_f64(&payload_buf[stream_offs[bi]..stream_offs[bi] + bs_sizes[bi]], j);
                                let norm = |v: f64, (lo, hi): (f64, f64)| -> u8 {
                                    let d = (hi - lo).abs().max(1e-9);
                                    (((v - lo) / d).clamp(0.0, 1.0) * 255.0).round() as u8
                                };
                                Some(egui::Color32::from_rgb(norm(r_v, rr), norm(g_v, gr), norm(b_v, br)))
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        out_chunk.push(PreviewPoint { x: wx, y: wy, z: wz, color });
                        total_added += 1;
                        got += 1;

                        if out_chunk.len() >= 1500 || (last_send.elapsed().as_millis() >= 80 && !out_chunk.is_empty()) {
                            let _ = tx.send(PreviewStreamMsg::Chunk(std::mem::take(&mut out_chunk)));
                            last_send = Instant::now();
                        }
                    }
                    deficit_pts = deficit_pts.saturating_sub(got);
                }
            }
        }

        cur += plen_pk;
        if cur % 4 != 0 {
            cur += 4 - (cur % 4);
        }
    }

    if !out_chunk.is_empty() {
        let _ = tx.send(PreviewStreamMsg::Chunk(out_chunk));
    }

    Ok(Some(total_added))
}

// Адаптивный расчёт стартового процента выборки по общему числу точек и формату файла
fn recommended_preview_percent(total_pts: usize, fmt: Option<PointCloudFormat>) -> f64 {
    if total_pts == 0 {
        return 100.0;
    }
    let target = match fmt {
        Some(PointCloudFormat::Las) => 250_000.0,
        _ => 500_000.0,
    };
    let raw_pct = (target / total_pts as f64) * 100.0;
    if raw_pct >= 100.0 {
        100.0
    } else if raw_pct >= 10.0 {
        raw_pct.round().clamp(10.0, 100.0)
    } else if raw_pct >= 1.0 {
        (raw_pct * 10.0).round() / 10.0
    } else if raw_pct >= 0.1 {
        (raw_pct * 100.0).round() / 100.0
    } else {
        ((raw_pct * 1000.0).round() / 1000.0).max(0.01)
    }
}

// Потоковое считывание заданного % точек для неблокирующего отображения облака с первых миллисекунд
fn sample_points_streaming(
    path: &Path,
    sample_percent: f64,
    cancel: Arc<AtomicBool>,
    tx: std::sync::mpsc::Sender<PreviewStreamMsg>,
) -> anyhow::Result<()> {
    sample_points_streaming_with_phase(path, sample_percent, 0, cancel, tx)
}

fn sample_points_streaming_with_phase(
    path: &Path,
    sample_percent: f64,
    phase_seed: usize,
    cancel: Arc<AtomicBool>,
    tx: std::sync::mpsc::Sender<PreviewStreamMsg>,
) -> anyhow::Result<()> {
    let pct_ratio = (sample_percent / 100.0).clamp(0.000_001, 1.0);
    let fmt = detect_format(path).unwrap_or(PointCloudFormat::Las);

    match fmt {
        PointCloudFormat::Las => {
            let mut reader = Reader::from_path(path)?;
            let total_pts = reader.header().number_of_points() as usize;
            if total_pts == 0 {
                let _ = tx.send(PreviewStreamMsg::Finished);
                return Ok(());
            }

            let target_samples = ((total_pts as f64) * pct_ratio).round().clamp(1.0, total_pts as f64) as usize;
            let step = (total_pts / target_samples).max(1);
            let start_i = if phase_seed == 0 { 0 } else { (phase_seed * 7) % step };

            const CHUNK_SIZE: usize = 2000;
            let mut chunk = Vec::with_capacity(CHUNK_SIZE);

            for i in (start_i..total_pts).step_by(step) {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                if reader.seek(i as u64).is_ok() {
                    if let Ok(Some(pt)) = reader.read_point() {
                        let color = pt.color.map(|c| {
                            let r = if c.red > 255 { (c.red >> 8) as u8 } else { c.red as u8 };
                            let g = if c.green > 255 { (c.green >> 8) as u8 } else { c.green as u8 };
                            let b = if c.blue > 255 { (c.blue >> 8) as u8 } else { c.blue as u8 };
                            egui::Color32::from_rgb(r, g, b)
                        });
                        chunk.push(PreviewPoint {
                            x: pt.x,
                            y: pt.y,
                            z: pt.z,
                            color,
                        });
                        if chunk.len() >= CHUNK_SIZE {
                            if tx.send(PreviewStreamMsg::Chunk(std::mem::take(&mut chunk))).is_err() {
                                return Ok(());
                            }
                        }
                    }
                }
            }
            if !chunk.is_empty() {
                let _ = tx.send(PreviewStreamMsg::Chunk(chunk));
            }
            let _ = tx.send(PreviewStreamMsg::Finished);
        }
        PointCloudFormat::E57 => {
            let mut reader = e57::E57Reader::from_file(path)?;
            let page_size = reader.header().page_size;
            let pcs = reader.pointclouds();
            let total_records: u64 = pcs.iter().map(|pc| pc.records).sum();
            let total_pts = total_records as usize;
            if total_pts == 0 {
                let _ = tx.send(PreviewStreamMsg::Finished);
                return Ok(());
            }

            if pcs.len() > 1 {
                // МНОГОСКАНОВЫЙ ПРОЕКТ E57 (наземное лазерное сканирование с множеством стоянок/панорам)
                // Считываем ВСЕ 100% стоянок волнами, пропорционально выбранному % от каждого скана
                let num_scans = pcs.len();
                let waves = num_scans.clamp(1, 8);
                let mut ordered_indices = Vec::with_capacity(num_scans);
                for w in 0..waves {
                    for i in (w..num_scans).step_by(waves) {
                        ordered_indices.push(i);
                    }
                }

                let io_threads = rayon::current_num_threads().clamp(2, 8);
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(io_threads)
                    .build()
                    .ok();

                let run_wave = |wave_indices: &[usize]| {
                    wave_indices.par_iter().for_each(|&idx| {
                        if cancel.load(Ordering::Relaxed) {
                            return;
                        }
                        let pc = &pcs[idx];
                        let target_pts = ((pc.records as f64) * pct_ratio).round().clamp(1.0, pc.records as f64) as usize;

                        if let Ok(mut paged) = PagedFastReader::open(path, page_size, 4) {
                            if let Ok(Some(_)) = fast_sample_e57_scan(&mut paged, pc, target_pts, phase_seed, &cancel, &tx) {
                                return;
                            }
                        }

                        // Безопасный fallback для нестандартной упаковки/сферических координат
                        if let Ok(mut r) = e57::E57Reader::from_file(path) {
                            let clouds = r.pointclouds();
                            if let Some(pc_ref) = clouds.get(idx) {
                                if let Ok(mut iter) = r.pointcloud_simple(pc_ref) {
                                    let has_cartesian = pc_ref.prototype.iter().any(|rec| rec.name == e57::RecordName::CartesianX);
                                    iter.spherical_to_cartesian(!has_cartesian);
                                    iter.cartesian_to_spherical(false);
                                    iter.intensity_to_color(false);
                                    iter.normalize_intensity(false);
                                    iter.apply_pose(true);

                                    let scan_step = (pc_ref.records / (target_pts as u64)).max(1) as usize;
                                    let mut chunk = Vec::with_capacity(target_pts);
                                    let mut p_idx = 0;
                                    while let Some(p) = iter.next() {
                                        if p_idx % 4096 == 0 && cancel.load(Ordering::Relaxed) {
                                            break;
                                        }
                                        if p_idx % scan_step == 0 {
                                            if let Ok(p) = p {
                                                if let e57::CartesianCoordinate::Valid { x, y, z } = p.cartesian {
                                                    let color = p.color.map(|c| {
                                                        let red = (c.red.clamp(0.0, 1.0) * 255.0).round() as u8;
                                                        let green = (c.green.clamp(0.0, 1.0) * 255.0).round() as u8;
                                                        let blue = (c.blue.clamp(0.0, 1.0) * 255.0).round() as u8;
                                                        egui::Color32::from_rgb(red, green, blue)
                                                    });
                                                    chunk.push(PreviewPoint { x, y, z, color });
                                                }
                                            }
                                        }
                                        p_idx += 1;
                                    }
                                    if !chunk.is_empty() {
                                        let _ = tx.send(PreviewStreamMsg::Chunk(chunk));
                                    }
                                }
                            }
                        }
                    });
                };

                // Разбиваем на батчи по волнам, чтобы UI сразу получал равномерный каркас по всему объекту
                let wave_chunk_size = (num_scans / waves).max(1);
                for wave_slice in ordered_indices.chunks(wave_chunk_size) {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Some(ref p) = pool {
                        p.install(|| run_wave(wave_slice));
                    } else {
                        run_wave(wave_slice);
                    }
                }
                let _ = tx.send(PreviewStreamMsg::Finished);
                return Ok(());
            }

            // ОДНОСКАНОВЫЙ E57 (одиночное плотное облако, например 2222.e57)
            let target_samples = ((total_pts as f64) * pct_ratio).round().clamp(1.0, total_pts as f64) as usize;
            if let Ok(mut paged) = PagedFastReader::open(path, page_size, 64) {
                if let Ok(Some(_)) = fast_sample_e57_scan(&mut paged, &pcs[0], target_samples, phase_seed, &cancel, &tx) {
                    let _ = tx.send(PreviewStreamMsg::Finished);
                    return Ok(());
                }
            }

            let step = (total_pts / target_samples).max(1);
            const CHUNK_SIZE: usize = 500;
            let mut chunk = Vec::with_capacity(CHUNK_SIZE);
            let mut total_sent = 0;
            let mut last_send = Instant::now();

            for pc in pcs {
                if cancel.load(Ordering::Relaxed) || total_sent >= target_samples {
                    break;
                }
                let mut iter = reader.pointcloud_simple(&pc)?;
                let has_cartesian = pc.prototype.iter().any(|r| r.name == e57::RecordName::CartesianX);
                iter.spherical_to_cartesian(!has_cartesian);
                iter.cartesian_to_spherical(false);
                iter.intensity_to_color(false);
                iter.normalize_intensity(false);
                iter.apply_pose(true);

                let mut idx = 0;
                while let Some(p) = iter.next() {
                    if idx % 4096 == 0 && cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    let p = p?;
                    if idx % step == 0 {
                        if let e57::CartesianCoordinate::Valid { x, y, z } = p.cartesian {
                            let color = p.color.map(|c| {
                                let r = (c.red.clamp(0.0, 1.0) * 255.0).round() as u8;
                                let g = (c.green.clamp(0.0, 1.0) * 255.0).round() as u8;
                                let b = (c.blue.clamp(0.0, 1.0) * 255.0).round() as u8;
                                egui::Color32::from_rgb(r, g, b)
                            });
                            chunk.push(PreviewPoint {
                                x,
                                y,
                                z,
                                color,
                            });
                            total_sent += 1;

                            if chunk.len() >= CHUNK_SIZE || (last_send.elapsed().as_millis() >= 80 && !chunk.is_empty()) {
                                if tx.send(PreviewStreamMsg::Chunk(std::mem::take(&mut chunk))).is_err() {
                                    return Ok(());
                                }
                                last_send = Instant::now();
                            }

                            if total_sent >= target_samples {
                                break;
                            }
                        }
                    }
                    idx += 1;
                }
            }
            if !chunk.is_empty() {
                let _ = tx.send(PreviewStreamMsg::Chunk(chunk));
            }
            let _ = tx.send(PreviewStreamMsg::Finished);
        }
    }

    Ok(())
}

// Быстрая синхронная выборка точек (для тестов и автономного анализа)
#[allow(dead_code)]
fn sample_points_from_file(path: &Path, sample_percent: f64) -> anyhow::Result<Vec<PreviewPoint>> {
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    sample_points_streaming(path, sample_percent, cancel, tx)?;
    let mut points = Vec::new();
    while let Ok(msg) = rx.recv() {
        match msg {
            PreviewStreamMsg::Chunk(mut c) => points.append(&mut c),
            PreviewStreamMsg::Finished => break,
            PreviewStreamMsg::Error(err) => return Err(anyhow::anyhow!(err)),
        }
    }
    Ok(points)
}

// Копирование фотопанорам и изображений (images2D) из исходного E57 в результирующий
fn copy_e57_images<T: std::io::Read + std::io::Seek, W: std::io::Read + std::io::Write + std::io::Seek>(
    reader: &mut e57::E57Reader<T>,
    writer: &mut e57::E57Writer<W>,
    guid_map: &HashMap<String, String>,
    scale_factor: f64,
) -> anyhow::Result<usize> {
    let images = reader.images();
    let mut copied_count = 0;

    for img in images {
        let img_guid = img.guid.as_deref().unwrap_or(&Uuid::new_v4().to_string()).to_string();
        let mut img_writer = writer.add_image(&img_guid)?;

        if let Some(name) = &img.name {
            img_writer.set_name(name);
        }
        if let Some(desc) = &img.description {
            img_writer.set_description(desc);
        }
        if let Some(vendor) = &img.sensor_vendor {
            img_writer.set_sensor_vendor(vendor);
        }
        if let Some(model) = &img.sensor_model {
            img_writer.set_sensor_model(model);
        }
        if let Some(serial) = &img.sensor_serial {
            img_writer.set_sensor_serial(serial);
        }
        if let Some(acq) = img.acquisition {
            img_writer.set_acquisition(acq);
        }

        // Обновление связки pointcloud_guid
        if let Some(old_pc_guid) = &img.pointcloud_guid {
            if let Some(new_pc_guid) = guid_map.get(old_pc_guid) {
                img_writer.set_pointcloud_guid(new_pc_guid);
            } else {
                img_writer.set_pointcloud_guid(old_pc_guid);
            }
        }

        // Трансформация камеры: масштабирование вектора смещения
        if let Some(mut transform) = img.transform {
            if scale_factor != 1.0 {
                transform.translation.x *= scale_factor;
                transform.translation.y *= scale_factor;
                transform.translation.z *= scale_factor;
            }
            img_writer.set_transform(transform);
        }

        // Превью VisualReference
        if let Some(vis_ref) = img.visual_reference {
            let mut buf = Vec::new();
            reader.blob(&vis_ref.blob.data, &mut buf)?;
            let mut cursor = std::io::Cursor::new(buf);
            let mut mask_buf = Vec::new();
            let mut mask_cursor = if let Some(mask_blob) = vis_ref.mask {
                reader.blob(&mask_blob, &mut mask_buf)?;
                Some(std::io::Cursor::new(mask_buf))
            } else {
                None
            };
            img_writer.add_visual_reference(
                vis_ref.blob.format,
                &mut cursor,
                vis_ref.properties,
                mask_cursor.as_mut().map(|c| c as &mut dyn std::io::Read),
            )?;
        }

        // Проекция фотопанорамы
        if let Some(projection) = img.projection {
            match projection {
                e57::Projection::Spherical(spherical) => {
                    let mut img_buf = Vec::new();
                    reader.blob(&spherical.blob.data, &mut img_buf)?;
                    let mut cursor = std::io::Cursor::new(img_buf);

                    let mut mask_buf = Vec::new();
                    let mut mask_cursor = if let Some(mask_blob) = spherical.mask {
                        reader.blob(&mask_blob, &mut mask_buf)?;
                        Some(std::io::Cursor::new(mask_buf))
                    } else {
                        None
                    };

                    img_writer.add_spherical(
                        spherical.blob.format,
                        &mut cursor,
                        spherical.properties,
                        mask_cursor.as_mut().map(|c| c as &mut dyn std::io::Read),
                    )?;
                }
                e57::Projection::Pinhole(pinhole) => {
                    let mut img_buf = Vec::new();
                    reader.blob(&pinhole.blob.data, &mut img_buf)?;
                    let mut cursor = std::io::Cursor::new(img_buf);

                    let mut mask_buf = Vec::new();
                    let mut mask_cursor = if let Some(mask_blob) = pinhole.mask {
                        reader.blob(&mask_blob, &mut mask_buf)?;
                        Some(std::io::Cursor::new(mask_buf))
                    } else {
                        None
                    };

                    img_writer.add_pinhole(
                        pinhole.blob.format,
                        &mut cursor,
                        pinhole.properties,
                        mask_cursor.as_mut().map(|c| c as &mut dyn std::io::Read),
                    )?;
                }
                e57::Projection::Cylindrical(cylindrical) => {
                    let mut img_buf = Vec::new();
                    reader.blob(&cylindrical.blob.data, &mut img_buf)?;
                    let mut cursor = std::io::Cursor::new(img_buf);

                    let mut mask_buf = Vec::new();
                    let mut mask_cursor = if let Some(mask_blob) = cylindrical.mask {
                        reader.blob(&mask_blob, &mut mask_buf)?;
                        Some(std::io::Cursor::new(mask_buf))
                    } else {
                        None
                    };

                    img_writer.add_cylindrical(
                        cylindrical.blob.format,
                        &mut cursor,
                        cylindrical.properties,
                        mask_cursor.as_mut().map(|c| c as &mut dyn std::io::Read),
                    )?;
                }
            }
        }

        img_writer.finalize()?;
        copied_count += 1;
    }

    Ok(copied_count)
}

// Вариант 1: Угловая строчно-колоночная децимация сетки E57 (k x k) с сохранением панорам и поддержкой LimitBox
fn process_structured_e57_to_e57<L, P>(
    input_path: &Path,
    output_path: &Path,
    grid_stride_k: usize,
    scale_factor: f64,
    use_thinning: bool,
    e57_use_f32: bool,
    limit_box: &LimitBox,
    log: &mut L,
    progress: &mut P,
) -> anyhow::Result<usize>
where
    L: FnMut(&str),
    P: FnMut(f32),
{
    let mut reader = e57::E57Reader::from_file(input_path)?;
    let total_points: u64 = reader.pointclouds().iter().map(|pc| pc.records).sum();
    let total_points_usize = total_points as usize;

    let k = if use_thinning { grid_stride_k.max(1) as i64 } else { 1i64 };

    log(&format!(
        "[Старт] Структурированное прореживание E57 (децимация сетки k = {})",
        k
    ));
    if k > 1 {
        log(&format!(
            "Ожидаемое уменьшение точек и размера файла: в {} раз (-{:.1}%)",
            k * k,
            (1.0 - 1.0 / ((k * k) as f64)) * 100.0
        ));
    }
    if limit_box.enabled {
        let (dx, dy, dz) = limit_box.dimensions();
        log(&format!(
            "Активна обрезка (Limit Box): X [{:.2}..{:.2}] ({:.2} м), Y [{:.2}..{:.2}] ({:.2} м), Z [{:.2}..{:.2}] ({:.2} м)",
            limit_box.min_x, limit_box.max_x, dx,
            limit_box.min_y, limit_box.max_y, dy,
            limit_box.min_z, limit_box.max_z, dz
        ));
    }

    let file_guid = Uuid::new_v4().to_string();
    let mut e57_writer = e57::E57Writer::from_file(output_path, &file_guid)?;

    let mut guid_map = HashMap::new();
    let mut written = 0;
    let mut total_processed = 0;
    let start_time = Instant::now();

    let pointclouds = reader.pointclouds();
    for (pc_idx, pc_meta) in pointclouds.iter().enumerate() {
        let scan_name = pc_meta.name.as_deref().unwrap_or("без имени");
        log(&format!(
            "Обработка структурированного скана {}/{} ({}, исходно ячеек в сетке: {})...",
            pc_idx + 1,
            pointclouds.len(),
            scan_name,
            pc_meta.records
        ));

        let has_color = pc_meta.has_color();
        let has_intensity = pc_meta.has_intensity();

        let has_invalid_state = pc_meta
            .prototype
            .iter()
            .any(|r| r.name == e57::RecordName::CartesianInvalidState);

        let orig_row_min = pc_meta
            .index_bounds
            .as_ref()
            .and_then(|ib| ib.row_min)
            .unwrap_or(0);
        let orig_col_min = pc_meta
            .index_bounds
            .as_ref()
            .and_then(|ib| ib.column_min)
            .unwrap_or(0);
        let out_row_max = pc_meta
            .index_bounds
            .as_ref()
            .and_then(|ib| ib.row_max)
            .map(|rm| ((rm - orig_row_min).max(0) / k).max(1))
            .unwrap_or(u32::MAX as i64);
        let out_col_max = pc_meta
            .index_bounds
            .as_ref()
            .and_then(|ib| ib.column_max)
            .map(|cm| ((cm - orig_col_min).max(0) / k).max(1))
            .unwrap_or(u32::MAX as i64);

        // Формирование прототипа
        let mut prototype = if e57_use_f32 {
            vec![
                e57::Record {
                    name: e57::RecordName::CartesianX,
                    data_type: e57::RecordDataType::Single { min: None, max: None },
                },
                e57::Record {
                    name: e57::RecordName::CartesianY,
                    data_type: e57::RecordDataType::Single { min: None, max: None },
                },
                e57::Record {
                    name: e57::RecordName::CartesianZ,
                    data_type: e57::RecordDataType::Single { min: None, max: None },
                },
            ]
        } else {
            vec![
                e57::Record::CARTESIAN_X_F64,
                e57::Record::CARTESIAN_Y_F64,
                e57::Record::CARTESIAN_Z_F64,
            ]
        };
        if has_invalid_state {
            prototype.push(e57::Record::CARTESIAN_INVALID_STATE);
        }

        // RowIndex / ColumnIndex с реальными границами скана для компактной упаковки E57
        prototype.push(e57::Record {
            name: e57::RecordName::RowIndex,
            data_type: e57::RecordDataType::Integer { min: 0, max: out_row_max },
        });
        prototype.push(e57::Record {
            name: e57::RecordName::ColumnIndex,
            data_type: e57::RecordDataType::Integer { min: 0, max: out_col_max },
        });

        if has_color {
            prototype.push(e57::Record::COLOR_RED_U8);
            prototype.push(e57::Record::COLOR_GREEN_U8);
            prototype.push(e57::Record::COLOR_BLUE_U8);
        }
        if has_intensity {
            prototype.push(e57::Record::INTENSITY_U16);
        }

        let new_pc_guid = Uuid::new_v4().to_string();
        if let Some(old_guid) = &pc_meta.guid {
            guid_map.insert(old_guid.clone(), new_pc_guid.clone());
        }

        let mut pc_writer = e57_writer.add_pointcloud(&new_pc_guid, prototype)?;

        // Метаданные скана
        if let Some(name) = &pc_meta.name {
            pc_writer.set_name(Some(name.clone()));
        }
        if let Some(desc) = &pc_meta.description {
            pc_writer.set_description(Some(desc.clone()));
        }
        if let Some(sensor_vendor) = &pc_meta.sensor_vendor {
            pc_writer.set_sensor_vendor(Some(sensor_vendor.clone()));
        }
        if let Some(sensor_model) = &pc_meta.sensor_model {
            pc_writer.set_sensor_model(Some(sensor_model.clone()));
        }
        if let Some(sensor_serial) = &pc_meta.sensor_serial {
            pc_writer.set_sensor_serial(Some(sensor_serial.clone()));
        }
        if let Some(acq_start) = &pc_meta.acquisition_start {
            pc_writer.set_acquisition_start(Some(acq_start.clone()));
        }
        if let Some(acq_end) = &pc_meta.acquisition_end {
            pc_writer.set_acquisition_end(Some(acq_end.clone()));
        }

        // Сохранение и масштабирование матрицы положения (pose)
        let scan_transform = pc_meta.transform.clone();
        if let Some(mut transform) = scan_transform.clone() {
            if scale_factor != 1.0 {
                transform.translation.x *= scale_factor;
                transform.translation.y *= scale_factor;
                transform.translation.z *= scale_factor;
            }
            pc_writer.set_transform(Some(transform));
        }

        if let Some(cl) = &pc_meta.color_limits {
            pc_writer.set_color_limits(Some(cl.clone()));
        }
        if let Some(il) = &pc_meta.intensity_limits {
            pc_writer.set_intensity_limits(Some(il.clone()));
        }

        // Читаем точки в локальных координатах сканера (apply_pose = false)
        let mut iter = reader.pointcloud_simple(pc_meta)?;
        iter.spherical_to_cartesian(true);
        iter.cartesian_to_spherical(false);
        iter.intensity_to_color(false);
        iter.apply_pose(false);

        let mut scan_points_written = 0usize;

        for p in iter {
            let p = p?;
            total_processed += 1;

            let row = p.row;
            let col = p.column;

            // Фильтрация по сетке k x k
            if k > 1 {
                let norm_row = row - orig_row_min;
                let norm_col = col - orig_col_min;
                if norm_row % k != 0 || norm_col % k != 0 {
                    continue; // Точка отсеивается и физически не пишется в файл
                }
            }

            let (local_x, local_y, local_z, invalid_state) = match p.cartesian {
                e57::CartesianCoordinate::Valid { x, y, z } => (x, y, z, 0i64),
                e57::CartesianCoordinate::Direction { x, y, z } => (x, y, z, 1i64),
                e57::CartesianCoordinate::Invalid => (0.0, 0.0, 0.0, 2i64),
            };

            // Проверка Limit Box в мировой системе координат
            if limit_box.enabled && invalid_state == 0 {
                let world_pt = transform_point((local_x, local_y, local_z), &scan_transform);
                if !limit_box.contains(world_pt.0, world_pt.1, world_pt.2) {
                    continue;
                }
            }

            // Новые непрерывные индексы децимированной сетки
            let new_row = (row - orig_row_min) / k;
            let new_col = (col - orig_col_min) / k;

            let out_x = if scale_factor != 1.0 { local_x * scale_factor } else { local_x };
            let out_y = if scale_factor != 1.0 { local_y * scale_factor } else { local_y };
            let out_z = if scale_factor != 1.0 { local_z * scale_factor } else { local_z };

            let mut values = Vec::with_capacity(9);
            if e57_use_f32 {
                values.push(e57::RecordValue::Single(out_x as f32));
                values.push(e57::RecordValue::Single(out_y as f32));
                values.push(e57::RecordValue::Single(out_z as f32));
            } else {
                values.push(e57::RecordValue::Double(out_x));
                values.push(e57::RecordValue::Double(out_y));
                values.push(e57::RecordValue::Double(out_z));
            }
            if has_invalid_state {
                values.push(e57::RecordValue::Integer(invalid_state));
            }
            values.push(e57::RecordValue::Integer(new_row));
            values.push(e57::RecordValue::Integer(new_col));

            if has_color {
                if let Some(c) = p.color {
                    let r = (c.red.clamp(0.0, 1.0) * 255.0).round() as i64;
                    let g = (c.green.clamp(0.0, 1.0) * 255.0).round() as i64;
                    let b = (c.blue.clamp(0.0, 1.0) * 255.0).round() as i64;
                    values.push(e57::RecordValue::Integer(r));
                    values.push(e57::RecordValue::Integer(g));
                    values.push(e57::RecordValue::Integer(b));
                } else {
                    values.push(e57::RecordValue::Integer(0));
                    values.push(e57::RecordValue::Integer(0));
                    values.push(e57::RecordValue::Integer(0));
                }
            }

            if has_intensity {
                if let Some(intensity) = p.intensity {
                    let val = (intensity.clamp(0.0, 1.0) * u16::MAX as f32).round() as i64;
                    values.push(e57::RecordValue::Integer(val));
                } else {
                    values.push(e57::RecordValue::Integer(0));
                }
            }

            pc_writer.add_point(values)?;
            written += 1;
            scan_points_written += 1;

            if total_points_usize > 0 && total_processed % (total_points_usize / 20).max(20_000) == 0 {
                let prog = (total_processed as f32 / total_points_usize as f32) * 85.0;
                progress(prog);
                log(&format!("Прогресс сканирования: {:.1}%", prog));
            }
        }

        pc_writer.finalize()?;
        log(&format!(
            "  -> Скан {} записан: {} ячеек сетки",
            pc_idx + 1,
            scan_points_written
        ));
    }

    // Копирование фотопанорам (images2D)
    log("Копирование фотопанорам и привязка к децимированным сканам...");
    progress(90.0);
    let copied_images = copy_e57_images(&mut reader, &mut e57_writer, &guid_map, scale_factor)?;
    log(&format!("Скопировано фотопанорам: {}", copied_images));

    progress(98.0);
    e57_writer.finalize()?;

    let duration = start_time.elapsed().as_secs_f32();
    log(&format!("Всего точек записано в результирующий файл: {}", format_number_with_spaces(written)));
    log(&format!(
        "[Конец] Децимация структурированного E57 завершена за {:.2} сек.",
        duration
    ));

    Ok(written)
}

// Обработка неструктурированного E57 -> E57 (Rayon batching)
fn process_unstructured_e57_to_e57<L, P>(
    input_path: &Path,
    output_path: &Path,
    voxel_size_m: f64,
    scale_factor: f64,
    use_thinning: bool,
    e57_use_f32: bool,
    limit_box: &LimitBox,
    log: &mut L,
    progress: &mut P,
) -> anyhow::Result<usize>
where
    L: FnMut(&str),
    P: FnMut(f32),
{
    let mut reader = e57::E57Reader::from_file(input_path)?;
    let total_points: u64 = reader.pointclouds().iter().map(|pc| pc.records).sum();
    let total_points_usize = total_points as usize;

    let has_color = reader.pointclouds().iter().any(|pc| pc.has_color());
    let has_intensity = reader.pointclouds().iter().any(|pc| pc.has_intensity());

    let mut prototype = if e57_use_f32 {
        vec![
            e57::Record {
                name: e57::RecordName::CartesianX,
                data_type: e57::RecordDataType::Single { min: None, max: None },
            },
            e57::Record {
                name: e57::RecordName::CartesianY,
                data_type: e57::RecordDataType::Single { min: None, max: None },
            },
            e57::Record {
                name: e57::RecordName::CartesianZ,
                data_type: e57::RecordDataType::Single { min: None, max: None },
            },
        ]
    } else {
        vec![
            e57::Record::CARTESIAN_X_F64,
            e57::Record::CARTESIAN_Y_F64,
            e57::Record::CARTESIAN_Z_F64,
        ]
    };
    if has_color {
        prototype.push(e57::Record::COLOR_RED_U8);
        prototype.push(e57::Record::COLOR_GREEN_U8);
        prototype.push(e57::Record::COLOR_BLUE_U8);
    }
    if has_intensity {
        prototype.push(e57::Record::INTENSITY_U16);
    }

    let file_guid = Uuid::new_v4().to_string();
    let mut e57_writer = e57::E57Writer::from_file(output_path, &file_guid)?;

    let pc_guid = Uuid::new_v4().to_string();
    let mut pc_writer = e57_writer.add_pointcloud(&pc_guid, prototype)?;

    let mut seen = RoaringBitmap::new();
    let mut written = 0;
    let mut processed = 0;
    let start_time = Instant::now();

    const BATCH_SIZE: usize = 65_536;
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    let pointclouds = reader.pointclouds();
    for (pc_idx, pc_meta) in pointclouds.iter().enumerate() {
        if pointclouds.len() > 1 {
            log(&format!("Чтение скана {}/{}...", pc_idx + 1, pointclouds.len()));
        }

        let mut iter = reader.pointcloud_simple(pc_meta)?;
        iter.spherical_to_cartesian(true);
        iter.cartesian_to_spherical(false);
        iter.intensity_to_color(false);
        iter.apply_pose(true);

        for p in iter {
            let p = p?;
            batch.push(p);

            if batch.len() >= BATCH_SIZE {
                let processed_chunk: Vec<Option<(Vec<e57::RecordValue>, u64)>> = batch
                    .par_iter()
                    .map(|p| {
                        let (mut x, mut y, mut z) = match p.cartesian {
                            e57::CartesianCoordinate::Valid { x, y, z } => (x, y, z),
                            _ => return None,
                        };

                        if limit_box.enabled && !limit_box.contains(x, y, z) {
                            return None;
                        }

                        if scale_factor != 1.0 {
                            x *= scale_factor;
                            y *= scale_factor;
                            z *= scale_factor;
                        }

                        let hash = if use_thinning && voxel_size_m > 0.0 {
                            let key = (
                                (x / voxel_size_m).round() as i64,
                                (y / voxel_size_m).round() as i64,
                                (z / voxel_size_m).round() as i64,
                            );
                            hash_voxel(&key)
                        } else {
                            0
                        };

                        let mut values = Vec::with_capacity(7);
                        if e57_use_f32 {
                            values.push(e57::RecordValue::Single(x as f32));
                            values.push(e57::RecordValue::Single(y as f32));
                            values.push(e57::RecordValue::Single(z as f32));
                        } else {
                            values.push(e57::RecordValue::Double(x));
                            values.push(e57::RecordValue::Double(y));
                            values.push(e57::RecordValue::Double(z));
                        }

                        if has_color {
                            if let Some(ref c) = p.color {
                                values.push(e57::RecordValue::Integer((c.red.clamp(0.0, 1.0) * 255.0).round() as i64));
                                values.push(e57::RecordValue::Integer((c.green.clamp(0.0, 1.0) * 255.0).round() as i64));
                                values.push(e57::RecordValue::Integer((c.blue.clamp(0.0, 1.0) * 255.0).round() as i64));
                            } else {
                                values.push(e57::RecordValue::Integer(0));
                                values.push(e57::RecordValue::Integer(0));
                                values.push(e57::RecordValue::Integer(0));
                            }
                        }

                        if has_intensity {
                            if let Some(intensity) = p.intensity {
                                values.push(e57::RecordValue::Integer((intensity.clamp(0.0, 1.0) * u16::MAX as f32).round() as i64));
                            } else {
                                values.push(e57::RecordValue::Integer(0));
                            }
                        }

                        Some((values, hash))
                    })
                    .collect();

                for item in processed_chunk {
                    if let Some((vals, hash)) = item {
                        if use_thinning && voxel_size_m > 0.0 {
                            let h32 = hash as u32;
                            if seen.contains(h32) {
                                continue;
                            }
                            seen.insert(h32);
                        }
                        pc_writer.add_point(vals)?;
                        written += 1;
                    }
                }

                processed += batch.len();
                batch.clear();

                if total_points_usize > 0 && processed % (total_points_usize / 20).max(10_000) < BATCH_SIZE {
                    let prog = (processed as f32 / total_points_usize as f32) * 90.0;
                    progress(prog);
                    log(&format!("Прогресс: {:.1}%", prog));
                }
            }
        }
    }

    if !batch.is_empty() {
        let processed_chunk: Vec<Option<(Vec<e57::RecordValue>, u64)>> = batch
            .par_iter()
            .map(|p| {
                let (mut x, mut y, mut z) = match p.cartesian {
                    e57::CartesianCoordinate::Valid { x, y, z } => (x, y, z),
                    _ => return None,
                };

                if limit_box.enabled && !limit_box.contains(x, y, z) {
                    return None;
                }

                if scale_factor != 1.0 {
                    x *= scale_factor;
                    y *= scale_factor;
                    z *= scale_factor;
                }

                let hash = if use_thinning && voxel_size_m > 0.0 {
                    let key = (
                        (x / voxel_size_m).round() as i64,
                        (y / voxel_size_m).round() as i64,
                        (z / voxel_size_m).round() as i64,
                    );
                    hash_voxel(&key)
                } else {
                    0
                };

                let mut values = Vec::with_capacity(7);
                if e57_use_f32 {
                    values.push(e57::RecordValue::Single(x as f32));
                    values.push(e57::RecordValue::Single(y as f32));
                    values.push(e57::RecordValue::Single(z as f32));
                } else {
                    values.push(e57::RecordValue::Double(x));
                    values.push(e57::RecordValue::Double(y));
                    values.push(e57::RecordValue::Double(z));
                }

                if has_color {
                    if let Some(ref c) = p.color {
                        values.push(e57::RecordValue::Integer((c.red.clamp(0.0, 1.0) * 255.0).round() as i64));
                        values.push(e57::RecordValue::Integer((c.green.clamp(0.0, 1.0) * 255.0).round() as i64));
                        values.push(e57::RecordValue::Integer((c.blue.clamp(0.0, 1.0) * 255.0).round() as i64));
                    } else {
                        values.push(e57::RecordValue::Integer(0));
                        values.push(e57::RecordValue::Integer(0));
                        values.push(e57::RecordValue::Integer(0));
                    }
                }

                if has_intensity {
                    if let Some(intensity) = p.intensity {
                        values.push(e57::RecordValue::Integer((intensity.clamp(0.0, 1.0) * u16::MAX as f32).round() as i64));
                    } else {
                        values.push(e57::RecordValue::Integer(0));
                    }
                }

                Some((values, hash))
            })
            .collect();

        for item in processed_chunk {
            if let Some((vals, hash)) = item {
                if use_thinning && voxel_size_m > 0.0 {
                    let h32 = hash as u32;
                    if seen.contains(h32) {
                        continue;
                    }
                    seen.insert(h32);
                }
                pc_writer.add_point(vals)?;
                written += 1;
            }
        }
    }

    pc_writer.finalize()?;

    // Перенос изображений, если они есть
    let mut guid_map = HashMap::new();
    for pc in reader.pointclouds() {
        if let Some(guid) = &pc.guid {
            guid_map.insert(guid.clone(), pc_guid.clone());
        }
    }
    let copied_images = copy_e57_images(&mut reader, &mut e57_writer, &guid_map, scale_factor)?;
    if copied_images > 0 {
        log(&format!("Скопировано фотопанорам: {}", copied_images));
    }

    e57_writer.finalize()?;

    let duration = start_time.elapsed().as_secs_f32();
    log(&format!("Записано точек: {}", format_number_with_spaces(written)));
    log(&format!("[Конец] Обработка завершена за {:.2} сек.", duration));

    Ok(written)
}

// Обработка LAS -> LAS (Rayon batching)
fn process_las_to_las<L, P>(
    input_path: &Path,
    output_path: &Path,
    voxel_size_m: f64,
    scale_factor: f64,
    use_thinning: bool,
    limit_box: &LimitBox,
    log: &mut L,
    progress: &mut P,
) -> anyhow::Result<usize>
where
    L: FnMut(&str),
    P: FnMut(f32),
{
    let mut reader = Reader::from_path(input_path)?;
    let header = reader.header().clone();
    let mut writer = Writer::from_path(output_path, header)?;

    let mut seen = RoaringBitmap::new();
    let mut written = 0;
    let total_points = reader.header().number_of_points() as usize;
    let mut processed = 0;

    let start_time = Instant::now();
    const BATCH_SIZE: usize = 65_536;
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    while let Some(point) = reader.read_point()? {
        batch.push(point);
        if batch.len() >= BATCH_SIZE {
            let processed_chunk: Vec<Option<(las::Point, u64)>> = batch
                .par_iter_mut()
                .map(|point| {
                    if limit_box.enabled && !limit_box.contains(point.x, point.y, point.z) {
                        return None;
                    }
                    if scale_factor != 1.0 {
                        point.x *= scale_factor;
                        point.y *= scale_factor;
                        point.z *= scale_factor;
                    }
                    let hash = if use_thinning && voxel_size_m > 0.0 {
                        let key = (
                            (point.x / voxel_size_m).round() as i64,
                            (point.y / voxel_size_m).round() as i64,
                            (point.z / voxel_size_m).round() as i64,
                        );
                        hash_voxel(&key)
                    } else {
                        0
                    };
                    Some((point.clone(), hash))
                })
                .collect();

            for item in processed_chunk {
                if let Some((pt, hash)) = item {
                    if use_thinning && voxel_size_m > 0.0 {
                        let h32 = hash as u32;
                        if seen.contains(h32) {
                            continue;
                        }
                        seen.insert(h32);
                    }
                    writer.write_point(pt)?;
                    written += 1;
                }
            }

            processed += batch.len();
            batch.clear();

            if total_points > 0 && processed % (total_points / 20).max(10_000) < BATCH_SIZE {
                let prog = (processed as f32 / total_points as f32) * 100.0;
                progress(prog);
                log(&format!("Прогресс: {:.1}%", prog));
            }
        }
    }

    if !batch.is_empty() {
        let processed_chunk: Vec<Option<(las::Point, u64)>> = batch
            .par_iter_mut()
            .map(|point| {
                if limit_box.enabled && !limit_box.contains(point.x, point.y, point.z) {
                    return None;
                }
                if scale_factor != 1.0 {
                    point.x *= scale_factor;
                    point.y *= scale_factor;
                    point.z *= scale_factor;
                }
                let hash = if use_thinning && voxel_size_m > 0.0 {
                    let key = (
                        (point.x / voxel_size_m).round() as i64,
                        (point.y / voxel_size_m).round() as i64,
                        (point.z / voxel_size_m).round() as i64,
                    );
                    hash_voxel(&key)
                } else {
                    0
                };
                Some((point.clone(), hash))
            })
            .collect();

        for item in processed_chunk {
            if let Some((pt, hash)) = item {
                if use_thinning && voxel_size_m > 0.0 {
                    let h32 = hash as u32;
                    if seen.contains(h32) {
                        continue;
                    }
                    seen.insert(h32);
                }
                writer.write_point(pt)?;
                written += 1;
            }
        }
    }

    let duration = start_time.elapsed().as_secs_f32();
    log(&format!("Записано точек: {}", format_number_with_spaces(written)));
    log(&format!("[Конец] Обработка завершена за {:.2} сек.", duration));

    Ok(written)
}

// Обработка LAS -> E57 (Rayon batching)
fn process_las_to_e57<L, P>(
    input_path: &Path,
    output_path: &Path,
    voxel_size_m: f64,
    scale_factor: f64,
    use_thinning: bool,
    e57_use_f32: bool,
    limit_box: &LimitBox,
    log: &mut L,
    progress: &mut P,
) -> anyhow::Result<usize>
where
    L: FnMut(&str),
    P: FnMut(f32),
{
    let mut reader = Reader::from_path(input_path)?;
    let total_points = reader.header().number_of_points() as usize;
    let has_color = reader.header().point_format().has_color;

    let file_guid = Uuid::new_v4().to_string();
    let mut e57_writer = e57::E57Writer::from_file(output_path, &file_guid)?;

    let mut prototype = if e57_use_f32 {
        vec![
            e57::Record {
                name: e57::RecordName::CartesianX,
                data_type: e57::RecordDataType::Single { min: None, max: None },
            },
            e57::Record {
                name: e57::RecordName::CartesianY,
                data_type: e57::RecordDataType::Single { min: None, max: None },
            },
            e57::Record {
                name: e57::RecordName::CartesianZ,
                data_type: e57::RecordDataType::Single { min: None, max: None },
            },
        ]
    } else {
        vec![
            e57::Record::CARTESIAN_X_F64,
            e57::Record::CARTESIAN_Y_F64,
            e57::Record::CARTESIAN_Z_F64,
        ]
    };
    if has_color {
        prototype.push(e57::Record::COLOR_RED_U8);
        prototype.push(e57::Record::COLOR_GREEN_U8);
        prototype.push(e57::Record::COLOR_BLUE_U8);
    }
    prototype.push(e57::Record::INTENSITY_U16);

    let pc_guid = Uuid::new_v4().to_string();
    let mut pc_writer = e57_writer.add_pointcloud(&pc_guid, prototype)?;

    let mut seen = RoaringBitmap::new();
    let mut written = 0;
    let mut processed = 0;
    let start_time = Instant::now();

    const BATCH_SIZE: usize = 65_536;
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    while let Some(point) = reader.read_point()? {
        batch.push(point);
        if batch.len() >= BATCH_SIZE {
            let processed_chunk: Vec<Option<(Vec<e57::RecordValue>, u64)>> = batch
                .par_iter_mut()
                .map(|point| {
                    if limit_box.enabled && !limit_box.contains(point.x, point.y, point.z) {
                        return None;
                    }
                    if scale_factor != 1.0 {
                        point.x *= scale_factor;
                        point.y *= scale_factor;
                        point.z *= scale_factor;
                    }
                    let hash = if use_thinning && voxel_size_m > 0.0 {
                        let key = (
                            (point.x / voxel_size_m).round() as i64,
                            (point.y / voxel_size_m).round() as i64,
                            (point.z / voxel_size_m).round() as i64,
                        );
                        hash_voxel(&key)
                    } else {
                        0
                    };

                    let mut record_values = Vec::with_capacity(7);
                    record_values.push(e57::RecordValue::Double(point.x));
                    record_values.push(e57::RecordValue::Double(point.y));
                    record_values.push(e57::RecordValue::Double(point.z));

                    if has_color {
                        if let Some(color) = point.color {
                            let r = if color.red > 255 { (color.red >> 8) as u8 } else { color.red as u8 };
                            let g = if color.green > 255 { (color.green >> 8) as u8 } else { color.green as u8 };
                            let b = if color.blue > 255 { (color.blue >> 8) as u8 } else { color.blue as u8 };
                            record_values.push(e57::RecordValue::Integer(r as i64));
                            record_values.push(e57::RecordValue::Integer(g as i64));
                            record_values.push(e57::RecordValue::Integer(b as i64));
                        } else {
                            record_values.push(e57::RecordValue::Integer(0));
                            record_values.push(e57::RecordValue::Integer(0));
                            record_values.push(e57::RecordValue::Integer(0));
                        }
                    }

                    record_values.push(e57::RecordValue::Integer(point.intensity as i64));
                    Some((record_values, hash))
                })
                .collect();

            for item in processed_chunk {
                if let Some((vals, hash)) = item {
                    if use_thinning && voxel_size_m > 0.0 {
                        let h32 = hash as u32;
                        if seen.contains(h32) {
                            continue;
                        }
                        seen.insert(h32);
                    }
                    pc_writer.add_point(vals)?;
                    written += 1;
                }
            }

            processed += batch.len();
            batch.clear();

            if total_points > 0 && processed % (total_points / 20).max(10_000) < BATCH_SIZE {
                let p = (processed as f32 / total_points as f32) * 100.0;
                progress(p);
                log(&format!("Прогресс: {:.1}%", p));
            }
        }
    }

    if !batch.is_empty() {
        let processed_chunk: Vec<Option<(Vec<e57::RecordValue>, u64)>> = batch
            .par_iter_mut()
            .map(|point| {
                if limit_box.enabled && !limit_box.contains(point.x, point.y, point.z) {
                    return None;
                }
                if scale_factor != 1.0 {
                    point.x *= scale_factor;
                    point.y *= scale_factor;
                    point.z *= scale_factor;
                }
                let hash = if use_thinning && voxel_size_m > 0.0 {
                    let key = (
                        (point.x / voxel_size_m).round() as i64,
                        (point.y / voxel_size_m).round() as i64,
                        (point.z / voxel_size_m).round() as i64,
                    );
                    hash_voxel(&key)
                } else {
                    0
                };

                let mut record_values = Vec::with_capacity(7);
                record_values.push(e57::RecordValue::Double(point.x));
                record_values.push(e57::RecordValue::Double(point.y));
                record_values.push(e57::RecordValue::Double(point.z));

                if has_color {
                    if let Some(color) = point.color {
                        let r = if color.red > 255 { (color.red >> 8) as u8 } else { color.red as u8 };
                        let g = if color.green > 255 { (color.green >> 8) as u8 } else { color.green as u8 };
                        let b = if color.blue > 255 { (color.blue >> 8) as u8 } else { color.blue as u8 };
                        record_values.push(e57::RecordValue::Integer(r as i64));
                        record_values.push(e57::RecordValue::Integer(g as i64));
                        record_values.push(e57::RecordValue::Integer(b as i64));
                    } else {
                        record_values.push(e57::RecordValue::Integer(0));
                        record_values.push(e57::RecordValue::Integer(0));
                        record_values.push(e57::RecordValue::Integer(0));
                    }
                }

                record_values.push(e57::RecordValue::Integer(point.intensity as i64));
                Some((record_values, hash))
            })
            .collect();

        for item in processed_chunk {
            if let Some((vals, hash)) = item {
                if use_thinning && voxel_size_m > 0.0 {
                    let h32 = hash as u32;
                    if seen.contains(h32) {
                        continue;
                    }
                    seen.insert(h32);
                }
                pc_writer.add_point(vals)?;
                written += 1;
            }
        }
    }

    pc_writer.finalize()?;
    e57_writer.finalize()?;

    let duration = start_time.elapsed().as_secs_f32();
    log(&format!("Записано точек: {}", format_number_with_spaces(written)));
    log(&format!("[Конец] Обработка завершена за {:.2} сек.", duration));

    Ok(written)
}

// Обработка E57 -> LAS (Rayon batching)
fn process_e57_to_las<L, P>(
    input_path: &Path,
    output_path: &Path,
    voxel_size_m: f64,
    scale_factor: f64,
    use_thinning: bool,
    limit_box: &LimitBox,
    log: &mut L,
    progress: &mut P,
) -> anyhow::Result<usize>
where
    L: FnMut(&str),
    P: FnMut(f32),
{
    let mut reader = e57::E57Reader::from_file(input_path)?;
    let total_points: u64 = reader.pointclouds().iter().map(|pc| pc.records).sum();
    let total_points_usize = total_points as usize;

    let has_color = reader.pointclouds().iter().any(|pc| pc.has_color());

    let mut builder = las::Builder::from((1, 2));
    builder.point_format = las::point::Format {
        has_color,
        ..Default::default()
    };

    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut min_z = f64::MAX;

    for pc in reader.pointclouds() {
        if let Some(bounds) = pc.get_cartesian_bounds() {
            if let Some(x) = bounds.x_min {
                if x < min_x { min_x = x; }
            }
            if let Some(y) = bounds.y_min {
                if y < min_y { min_y = y; }
            }
            if let Some(z) = bounds.z_min {
                if z < min_z { min_z = z; }
            }
        }
    }

    builder.transforms.x.scale = 0.0001;
    builder.transforms.y.scale = 0.0001;
    builder.transforms.z.scale = 0.0001;
    if min_x < f64::MAX && min_y < f64::MAX && min_z < f64::MAX {
        builder.transforms.x.offset = min_x * scale_factor;
        builder.transforms.y.offset = min_y * scale_factor;
        builder.transforms.z.offset = min_z * scale_factor;
    }

    let header = builder.into_header()?;
    let mut writer = Writer::from_path(output_path, header)?;

    let mut seen = RoaringBitmap::new();
    let mut written = 0;
    let mut processed = 0;
    let start_time = Instant::now();

    const BATCH_SIZE: usize = 65_536;
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    let pointclouds = reader.pointclouds();
    for (pc_idx, pc_meta) in pointclouds.iter().enumerate() {
        if pointclouds.len() > 1 {
            log(&format!("Чтение скана {}/{}...", pc_idx + 1, pointclouds.len()));
        }

        let mut iter = reader.pointcloud_simple(pc_meta)?;
        iter.spherical_to_cartesian(true);
        iter.cartesian_to_spherical(false);
        iter.intensity_to_color(false);
        iter.apply_pose(true);

        for p in iter {
            let p = p?;
            batch.push(p);

            if batch.len() >= BATCH_SIZE {
                let processed_chunk: Vec<Option<(las::Point, u64)>> = batch
                    .par_iter()
                    .map(|p| {
                        let (mut x, mut y, mut z) = match p.cartesian {
                            e57::CartesianCoordinate::Valid { x, y, z } => (x, y, z),
                            _ => return None,
                        };

                        if limit_box.enabled && !limit_box.contains(x, y, z) {
                            return None;
                        }

                        if scale_factor != 1.0 {
                            x *= scale_factor;
                            y *= scale_factor;
                            z *= scale_factor;
                        }

                        let hash = if use_thinning && voxel_size_m > 0.0 {
                            let key = (
                                (x / voxel_size_m).round() as i64,
                                (y / voxel_size_m).round() as i64,
                                (z / voxel_size_m).round() as i64,
                            );
                            hash_voxel(&key)
                        } else {
                            0
                        };

                        let mut las_point = las::Point {
                            x,
                            y,
                            z,
                            ..Default::default()
                        };

                        if let Some(ref color) = p.color {
                            las_point.color = Some(las::Color {
                                red: (color.red.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16,
                                green: (color.green.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16,
                                blue: (color.blue.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16,
                            });
                        }

                        if let Some(intensity) = p.intensity {
                            las_point.intensity = (intensity.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16;
                        }

                        Some((las_point, hash))
                    })
                    .collect();

                for item in processed_chunk {
                    if let Some((pt, hash)) = item {
                        if use_thinning && voxel_size_m > 0.0 {
                            let h32 = hash as u32;
                            if seen.contains(h32) {
                                continue;
                            }
                            seen.insert(h32);
                        }
                        writer.write_point(pt)?;
                        written += 1;
                    }
                }

                processed += batch.len();
                batch.clear();

                if total_points_usize > 0 && processed % (total_points_usize / 20).max(10_000) < BATCH_SIZE {
                    let prog = (processed as f32 / total_points_usize as f32) * 100.0;
                    progress(prog);
                    log(&format!("Прогресс: {:.1}%", prog));
                }
            }
        }
    }

    if !batch.is_empty() {
        let processed_chunk: Vec<Option<(las::Point, u64)>> = batch
            .par_iter()
            .map(|p| {
                let (mut x, mut y, mut z) = match p.cartesian {
                    e57::CartesianCoordinate::Valid { x, y, z } => (x, y, z),
                    _ => return None,
                };

                if limit_box.enabled && !limit_box.contains(x, y, z) {
                    return None;
                }

                if scale_factor != 1.0 {
                    x *= scale_factor;
                    y *= scale_factor;
                    z *= scale_factor;
                }

                let hash = if use_thinning && voxel_size_m > 0.0 {
                    let key = (
                        (x / voxel_size_m).round() as i64,
                        (y / voxel_size_m).round() as i64,
                        (z / voxel_size_m).round() as i64,
                    );
                    hash_voxel(&key)
                } else {
                    0
                };

                let mut las_point = las::Point {
                    x,
                    y,
                    z,
                    ..Default::default()
                };

                if let Some(ref color) = p.color {
                    las_point.color = Some(las::Color {
                        red: (color.red.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16,
                        green: (color.green.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16,
                        blue: (color.blue.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16,
                    });
                }

                if let Some(intensity) = p.intensity {
                    las_point.intensity = (intensity.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16;
                }

                Some((las_point, hash))
            })
            .collect();

        for item in processed_chunk {
            if let Some((pt, hash)) = item {
                if use_thinning && voxel_size_m > 0.0 {
                    let h32 = hash as u32;
                    if seen.contains(h32) {
                        continue;
                    }
                    seen.insert(h32);
                }
                writer.write_point(pt)?;
                written += 1;
            }
        }
    }

    let duration = start_time.elapsed().as_secs_f32();
    log(&format!("Записано точек: {}", format_number_with_spaces(written)));
    log(&format!("[Конец] Обработка завершена за {:.2} сек.", duration));

    Ok(written)
}

// Диспетчер обработки файла
#[allow(clippy::too_many_arguments)]
fn process_point_cloud<L, P>(
    input_path: &Path,
    output_path: &Path,
    voxel_size_m: f64,
    scale_factor: f64,
    use_thinning: bool,
    grid_stride_k: usize,
    thinning_method: ThinningMethod,
    e57_use_f32: bool,
    limit_box: &LimitBox,
    log: &mut L,
    progress: &mut P,
) -> anyhow::Result<usize>
where
    L: FnMut(&str),
    P: FnMut(f32),
{
    let in_fmt = detect_format(input_path).unwrap_or_else(|| {
        if e57::E57Reader::from_file(input_path).is_ok() {
            PointCloudFormat::E57
        } else {
            PointCloudFormat::Las
        }
    });

    let out_fmt = detect_format(output_path).unwrap_or(in_fmt);

    log(&format!(
        "Многопоточная оптимизация Rayon: выделено рабочих потоков: {}",
        rayon::current_num_threads()
    ));

    if limit_box.enabled {
        let (dx, dy, dz) = limit_box.dimensions();
        log(&format!(
            "Фильтр Limit Box: X [{:.2}..{:.2}] ({:.2} м), Y [{:.2}..{:.2}] ({:.2} м), Z [{:.2}..{:.2}] ({:.2} м) | Объем: {:.1} куб. м",
            limit_box.min_x, limit_box.max_x, dx,
            limit_box.min_y, limit_box.max_y, dy,
            limit_box.min_z, limit_box.max_z, dz,
            limit_box.volume()
        ));
    }

    match (in_fmt, out_fmt) {
        (PointCloudFormat::Las, PointCloudFormat::Las) => {
            process_las_to_las(input_path, output_path, voxel_size_m, scale_factor, use_thinning, limit_box, log, progress)
        }
        (PointCloudFormat::Las, PointCloudFormat::E57) => {
            process_las_to_e57(input_path, output_path, voxel_size_m, scale_factor, use_thinning, e57_use_f32, limit_box, log, progress)
        }
        (PointCloudFormat::E57, PointCloudFormat::E57) => {
            let is_structured = if let Ok(reader) = e57::E57Reader::from_file(input_path) {
                reader.pointclouds().iter().any(|pc| pc.has_row_column())
            } else {
                false
            };

            if is_structured && thinning_method == ThinningMethod::GridStride {
                log(&format!(
                    "Инициализирован режим структурированного E57 (децимация сетки k x k + фотопанорамы, координаты {})...",
                    if e57_use_f32 { "f32" } else { "f64" }
                ));
                process_structured_e57_to_e57(input_path, output_path, grid_stride_k, scale_factor, use_thinning, e57_use_f32, limit_box, log, progress)
            } else {
                if is_structured {
                    log(&format!(
                        "Инициализирован режим прореживания структурированного E57 по среднему расстоянию (3D-сетка вокселей, координаты {})...",
                        if e57_use_f32 { "f32" } else { "f64" }
                    ));
                } else {
                    log(&format!(
                        "Инициализирован режим неструктурированного E57 (3D-сетка вокселей, координаты {})...",
                        if e57_use_f32 { "f32" } else { "f64" }
                    ));
                }
                process_unstructured_e57_to_e57(input_path, output_path, voxel_size_m, scale_factor, use_thinning, e57_use_f32, limit_box, log, progress)
            }
        }
        (PointCloudFormat::E57, PointCloudFormat::Las) => {
            process_e57_to_las(input_path, output_path, voxel_size_m, scale_factor, use_thinning, limit_box, log, progress)
        }
    }
}

// Сообщения из фонового рабочего потока
enum WorkerMsg {
    Log(String),
    Progress(f32),
    Finished {
        result_point_count: usize,
        elapsed: Duration,
        scale: f64,
    },
    Error(String),
}

// Структура GUI приложения
struct ThinLasApp {
    voxel_size: f64,
    e57_use_f32: bool,
    scale_factor: f64,
    use_thinning: bool,
    use_scaling: bool,
    grid_stride_k: usize,
    thinning_method: ThinningMethod,
    limit_box: LimitBox,
    viewport_3d: Viewport3D,
    input_file: Option<PathBuf>,
    output_file: Option<PathBuf>,
    source_file_size: u64,
    sample_points: Vec<PreviewPoint>,
    cached_eval_voxel_size: f64,
    cached_reduction_pct: f64,
    cached_est_bytes: u64,
    cloud_surface_area: Option<f64>,
    cloud_bounds_min: Option<(f64, f64, f64)>,
    cloud_bounds_max: Option<(f64, f64, f64)>,
    density_eval_status: Option<String>,
    log: String,
    source_version: Option<String>,
    source_point_count: usize,
    result_point_count: usize,
    progress: f32,
    elapsed_time: Duration,
    is_processing: bool,
    worker_rx: Option<Receiver<WorkerMsg>>,
    is_loading_preview: bool,
    preview_rx: Option<Receiver<PreviewStreamMsg>>,
    preview_cancel: Option<Arc<AtomicBool>>,
    cached_preview_path: Option<PathBuf>,
    cached_preview_size: u64,
    preview_sample_percent: f64,
    cached_preview_percent: f64,
    preview_phase: usize,
}

impl Default for ThinLasApp {
    fn default() -> Self {
        Self {
            voxel_size: 0.05,
            e57_use_f32: false,
            scale_factor: 1.0,
            use_thinning: false,
            use_scaling: false,
            grid_stride_k: 2,
            thinning_method: ThinningMethod::VoxelGrid,
            limit_box: LimitBox::default(),
            viewport_3d: Viewport3D::default(),
            input_file: None,
            output_file: None,
            source_file_size: 0,
            sample_points: Vec::new(),
            cached_eval_voxel_size: 0.0,
            cached_reduction_pct: 0.0,
            cached_est_bytes: 0,
            cloud_surface_area: None,
            cloud_bounds_min: None,
            cloud_bounds_max: None,
            density_eval_status: None,
            log: String::new(),
            source_version: None,
            source_point_count: 0,
            result_point_count: 0,
            progress: 0.0,
            elapsed_time: Duration::from_secs(0),
            is_processing: false,
            worker_rx: None,
            is_loading_preview: false,
            preview_rx: None,
            preview_cancel: None,
            cached_preview_path: None,
            cached_preview_size: 0,
            preview_sample_percent: 1.0,
            cached_preview_percent: 0.0,
            preview_phase: 0,
        }
    }
}

impl ThinLasApp {
    // Инкрементальное обновление точек при смене % выборки:
    // • при увеличении % — старые точки сохраняются, в фоне догружается только недостающая дельта
    // • при уменьшении % — прореживаем уже загруженные точки в RAM мгновенно (0 мс)
    fn reload_preview_points(&mut self) {
        let new_pct = self.preview_sample_percent.clamp(0.001, 100.0);
        let old_pct = if self.cached_preview_percent > 0.0 {
            self.cached_preview_percent
        } else {
            new_pct
        };

        if !self.sample_points.is_empty()
            && self.cached_preview_path == self.input_file
            && self.cached_preview_size == self.source_file_size
        {
            if (new_pct - old_pct).abs() < 1e-6 && !self.is_loading_preview {
                return;
            }
            if new_pct < old_pct {
                // Останавливаем возможную дозагрузку и мгновенно прореживаем в RAM без чтения с диска
                if let Some(ref c) = self.preview_cancel {
                    c.store(true, Ordering::Relaxed);
                }
                self.preview_cancel = None;
                self.preview_rx = None;
                self.is_loading_preview = false;

                let keep_ratio = (new_pct / old_pct).clamp(0.0001, 1.0);
                let target_len = ((self.sample_points.len() as f64) * keep_ratio).round().max(1.0) as usize;
                let mut kept = Vec::with_capacity(target_len);
                let mut acc = 0.0f64;
                for pt in &self.sample_points {
                    acc += keep_ratio;
                    if acc >= 1.0 {
                        acc -= 1.0;
                        kept.push(pt.clone());
                    }
                }
                if kept.is_empty() && !self.sample_points.is_empty() {
                    kept.push(self.sample_points[0].clone());
                }
                self.sample_points = kept;
                self.cached_preview_percent = new_pct;
                return;
            } else if !self.is_loading_preview {
                if let Some(ref path) = self.input_file {
                    // Увеличение %: НЕ очищаем экран — догружаем только недостающую разницу со сдвигом фазы
                    let delta_pct = (new_pct - old_pct).clamp(0.001, 100.0);
                    self.is_loading_preview = true;
                    self.preview_phase = self.preview_phase.wrapping_add(1).max(1);
                    let phase = self.preview_phase;
                    let path_clone = path.clone();
                    let (tx, rx) = std::sync::mpsc::channel();
                    self.preview_rx = Some(rx);
                    let cancel = Arc::new(AtomicBool::new(false));
                    self.preview_cancel = Some(cancel.clone());

                    std::thread::spawn(move || {
                        let tx_err = tx.clone();
                        if let Err(err) = sample_points_streaming_with_phase(&path_clone, delta_pct, phase, cancel, tx) {
                            let _ = tx_err.send(PreviewStreamMsg::Error(err.to_string()));
                        }
                    });
                    return;
                }
            }
        }

        if let Some(ref c) = self.preview_cancel {
            c.store(true, Ordering::Relaxed);
        }
        self.preview_cancel = None;
        self.preview_rx = None;
        self.is_loading_preview = false;
        self.cached_preview_path = None;
        self.cached_preview_size = 0;
        self.cached_preview_percent = 0.0;
        self.preview_phase = 0;
        self.sample_points.clear();
        self.ensure_preview_points();
    }

    // Асинхронный запуск фонового потокового считывания точек для 3D-предпросмотра
    fn ensure_preview_points(&mut self) {
        // Проверка RAM-кэша: если для этого файла и % выборка уже в памяти — открываем мгновенно
        if let Some(ref path) = self.input_file {
            if self.cached_preview_path.as_ref() == Some(path)
                && self.cached_preview_size == self.source_file_size
                && (self.cached_preview_percent - self.preview_sample_percent).abs() < 1e-6
                && !self.sample_points.is_empty()
            {
                return;
            }
        }

        if !self.is_loading_preview {
            if let Some(ref path) = self.input_file {
                self.is_loading_preview = true;
                self.sample_points.clear();
                let path_clone = path.clone();
                let (tx, rx) = std::sync::mpsc::channel();
                self.preview_rx = Some(rx);
                let cancel = Arc::new(AtomicBool::new(false));
                self.preview_cancel = Some(cancel.clone());
                let sample_pct = self.preview_sample_percent.clamp(0.001, 100.0);

                std::thread::spawn(move || {
                    let tx_err = tx.clone();
                    if let Err(err) = sample_points_streaming(&path_clone, sample_pct, cancel, tx) {
                        let _ = tx_err.send(PreviewStreamMsg::Error(err.to_string()));
                    }
                });
            }
        }
    }

    // Централизованная мгновенная загрузка файла (для кнопки диалога и Drag & Drop)
    fn load_input_file(&mut self, path: PathBuf) {
        if let Some(ref c) = self.preview_cancel {
            c.store(true, Ordering::Relaxed);
        }
        self.preview_cancel = None;
        self.preview_rx = None;
        self.is_loading_preview = false;
        self.cached_preview_path = None;
        self.cached_preview_size = 0;
        self.cached_preview_percent = 0.0;
        self.input_file = Some(path.clone());
        self.source_version = None;
        self.source_point_count = 0;
        self.source_file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        self.sample_points.clear();
        self.cached_eval_voxel_size = 0.0;
        self.cached_reduction_pct = 0.0;
        self.cached_est_bytes = 0;
        self.cloud_surface_area = None;
        self.cloud_bounds_min = None;
        self.cloud_bounds_max = None;
        self.density_eval_status = None;
        self.e57_use_f32 = false;

        let format_hint = detect_format(&path);
        let mut min_x = f64::MAX; let mut max_x = f64::MIN;
        let mut min_y = f64::MAX; let mut max_y = f64::MIN;
        let mut min_z = f64::MAX; let mut max_z = f64::MIN;
        let mut has_bounds = false;

        // Быстрая проверка E57 по заголовку XML без чтения точек
        if format_hint == Some(PointCloudFormat::E57) || format_hint.is_none() {
            if let Ok(reader) = e57::E57Reader::from_file(&path) {
                let pc_count = reader.pointclouds().len();
                let image_count = reader.images().len();
                let is_structured = reader.pointclouds().iter().any(|pc| pc.has_row_column());
                self.e57_use_f32 = is_structured;
                let total_points: u64 = reader.pointclouds().iter().map(|pc| pc.records).sum();

                for pc in reader.pointclouds() {
                    if let Some(bounds) = pc.get_cartesian_bounds() {
                        if let (Some(x0), Some(x1), Some(y0), Some(y1), Some(z0), Some(z1)) = (
                            bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max, bounds.z_min, bounds.z_max
                        ) {
                            let center_x = (x0 + x1) / 2.0;
                            let center_y = (y0 + y1) / 2.0;
                            let is_already_world = if let Some(ref t) = pc.transform {
                                let dist_to_trans = ((center_x - t.translation.x).powi(2) + (center_y - t.translation.y).powi(2)).sqrt();
                                let dist_to_zero = (center_x.powi(2) + center_y.powi(2)).sqrt();
                                dist_to_trans < dist_to_zero
                            } else {
                                true
                            };

                            let corners = [
                                (x0, y0, z0), (x0, y0, z1), (x0, y1, z0), (x0, y1, z1),
                                (x1, y0, z0), (x1, y0, z1), (x1, y1, z0), (x1, y1, z1),
                            ];
                            for (cx, cy, cz) in corners {
                                let (wx, wy, wz) = if is_already_world {
                                    (cx, cy, cz)
                                } else {
                                    transform_point((cx, cy, cz), &pc.transform)
                                };
                                min_x = min_x.min(wx); max_x = max_x.max(wx);
                                min_y = min_y.min(wy); max_y = max_y.max(wy);
                                min_z = min_z.min(wz); max_z = max_z.max(wz);
                            }
                            has_bounds = true;
                        }
                    }
                }

                let version = if is_structured {
                    format!("E57 (структурированный, сканов: {}, панорам: {})", pc_count, image_count)
                } else if image_count > 0 {
                    format!("E57 (сканов: {}, панорам: {})", pc_count, image_count)
                } else if pc_count > 1 {
                    format!("E57 (сканов: {})", pc_count)
                } else {
                    "E57".to_string()
                };
                self.source_version = Some(version);
                self.source_point_count = total_points as usize;
                if is_structured {
                    self.thinning_method = ThinningMethod::GridStride;
                } else {
                    self.thinning_method = ThinningMethod::VoxelGrid;
                }
            }
        }

        // Быстрая проверка LAS по бинарному заголовку
        if self.source_version.is_none() {
            if let Ok(reader) = Reader::from_path(&path) {
                let version = format!(
                    "LAS {}.{}",
                    reader.header().version().major,
                    reader.header().version().minor
                );
                let b = reader.header().bounds();
                min_x = b.min.x; max_x = b.max.x;
                min_y = b.min.y; max_y = b.max.y;
                min_z = b.min.z; max_z = b.max.z;
                has_bounds = max_x > min_x && max_y > min_y;

                self.source_version = Some(version);
                self.source_point_count = reader.header().number_of_points() as usize;
                self.thinning_method = ThinningMethod::VoxelGrid;
            }
        }

        if has_bounds && max_x > min_x && max_y > min_y {
            let dx = (max_x - min_x).max(1.0);
            let dy = (max_y - min_y).max(1.0);
            let dz = (max_z - min_z).abs().max(0.1);
            let s_box = 2.0 * (dx * dy + dx * dz + dy * dz);
            self.cloud_surface_area = Some(s_box);
            self.cloud_bounds_min = Some((min_x, min_y, min_z));
            self.cloud_bounds_max = Some((max_x, max_y, max_z));

            // Инициализация параметров LimitBox
            self.limit_box.min_x = min_x;
            self.limit_box.max_x = max_x;
            self.limit_box.min_y = min_y;
            self.limit_box.max_y = max_y;
            self.limit_box.min_z = min_z;
            self.limit_box.max_z = max_z;

            // Центрирование камеры в 3D вьювере
            self.viewport_3d.cloud_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0);
            self.viewport_3d.cloud_extent = (dx, dy, dz);
            self.viewport_3d.bounds_min = (min_x, min_y, min_z);
            self.viewport_3d.bounds_max = (max_x, max_y, max_z);
            let diag = (dx * dx + dy * dy + dz * dz).sqrt().max(10.0);
            self.viewport_3d.distance = (diag * 1.3) as f32;
            self.viewport_3d.pan_offset = egui::Vec2::ZERO;
            self.viewport_3d.yaw = -0.785;
            self.viewport_3d.pitch = 0.52;
        }

        // Адаптивный стартовый % выборки по общему числу точек и формату
        self.preview_sample_percent = recommended_preview_percent(self.source_point_count, format_hint);

        // Автозаполнение имени выходного файла
        let default_ext = if detect_format(&path) == Some(PointCloudFormat::E57) { "e57" } else { "las" };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        let mut out = path.clone();
        out.set_file_name(format!("{}_processed.{}", stem, default_ext));
        self.output_file = Some(out);
    }
}

impl eframe::App for ThinLasApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Прием сообщений от фонового потока
        if let Some(ref rx) = self.worker_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    WorkerMsg::Log(line) => {
                        self.log.push_str(&format!("{}\n", line));
                    }
                    WorkerMsg::Progress(p) => {
                        self.progress = p;
                    }
                    WorkerMsg::Finished {
                        result_point_count,
                        elapsed,
                        scale,
                    } => {
                        self.result_point_count = result_point_count;
                        self.elapsed_time = elapsed;
                        self.progress = 100.0;
                        self.log.push_str("Обработка завершена.\n");

                        if scale != 1.0 {
                            self.log.push_str(&format!(
                                "Применен коэффициент масштабирования: {:.3}x\n",
                                scale
                            ));
                        }
                        self.is_processing = false;
                    }
                    WorkerMsg::Error(err) => {
                        self.log.push_str(&format!("Ошибка: {}\n", err));
                        self.is_processing = false;
                    }
                }
            }
            if !self.is_processing {
                self.worker_rx = None;
            }
        }

        // Прием потоковых точек 3D-предпросмотра из фонового потока
        let mut preview_done = false;
        if let Some(ref rx) = self.preview_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    PreviewStreamMsg::Chunk(mut chunk) => {
                        let was_empty = self.sample_points.is_empty();
                        if !chunk.is_empty() {
                            let mut c_min_x = f64::MAX; let mut c_max_x = f64::MIN;
                            let mut c_min_y = f64::MAX; let mut c_max_y = f64::MIN;
                            let mut c_min_z = f64::MAX; let mut c_max_z = f64::MIN;
                            for p in &chunk {
                                c_min_x = c_min_x.min(p.x); c_max_x = c_max_x.max(p.x);
                                c_min_y = c_min_y.min(p.y); c_max_y = c_max_y.max(p.y);
                                c_min_z = c_min_z.min(p.z); c_max_z = c_max_z.max(p.z);
                            }

                            // Проверяем, совпадают ли уже заданные габариты с реальными точками
                            let needs_reset = match (self.cloud_bounds_min, self.cloud_bounds_max) {
                                (None, _) | (_, None) => true,
                                (Some((bx0, by0, bz0)), Some((bx1, by1, bz1))) => {
                                    let diag = ((bx1 - bx0).powi(2) + (by1 - by0).powi(2) + (bz1 - bz0).powi(2)).sqrt().max(10.0);
                                    c_max_x < bx0 - diag || c_min_x > bx1 + diag
                                        || c_max_y < by0 - diag || c_min_y > by1 + diag
                                }
                            };

                            if needs_reset && was_empty {
                                self.cloud_bounds_min = Some((c_min_x, c_min_y, c_min_z));
                                self.cloud_bounds_max = Some((c_max_x, c_max_y, c_max_z));
                            } else if let (Some(bmin), Some(bmax)) = (self.cloud_bounds_min.as_mut(), self.cloud_bounds_max.as_mut()) {
                                bmin.0 = bmin.0.min(c_min_x); bmin.1 = bmin.1.min(c_min_y); bmin.2 = bmin.2.min(c_min_z);
                                bmax.0 = bmax.0.max(c_max_x); bmax.1 = bmax.1.max(c_max_y); bmax.2 = bmax.2.max(c_max_z);
                            }

                            if let (Some((min_x, min_y, min_z)), Some((max_x, max_y, max_z))) = (self.cloud_bounds_min, self.cloud_bounds_max) {
                                if max_x > min_x && max_y > min_y {
                                    if !self.limit_box.enabled && (needs_reset || was_empty) {
                                        self.limit_box.reset_to_bounds((min_x, min_y, min_z), (max_x, max_y, max_z));
                                    }
                                    if needs_reset || was_empty {
                                        self.viewport_3d.cloud_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0);
                                        let dx = (max_x - min_x).max(1.0);
                                        let dy = (max_y - min_y).max(1.0);
                                        let dz = (max_z - min_z).max(1.0);
                                        self.viewport_3d.cloud_extent = (dx, dy, dz);
                                        self.viewport_3d.bounds_min = (min_x, min_y, min_z);
                                        self.viewport_3d.bounds_max = (max_x, max_y, max_z);
                                        let diag = (dx * dx + dy * dy + dz * dz).sqrt().max(10.0);
                                        self.viewport_3d.distance = (diag * 1.3) as f32;
                                    }
                                }
                            }
                            self.sample_points.append(&mut chunk);
                        }
                        ctx.request_repaint();
                    }
                    PreviewStreamMsg::Finished => {
                        self.is_loading_preview = false;
                        self.cached_preview_path = self.input_file.clone();
                        self.cached_preview_size = self.source_file_size;
                        self.cached_preview_percent = self.preview_sample_percent;
                        let count = self.sample_points.len();
                        if let Some(ref st) = self.density_eval_status {
                            if st.starts_with("Запущено") {
                                self.density_eval_status = Some(format!(
                                    "Оценка выполнена по выборке из {} точек",
                                    format_number_with_spaces(count)
                                ));
                            }
                        }
                        preview_done = true;
                        ctx.request_repaint();
                    }
                    PreviewStreamMsg::Error(e) => {
                        self.is_loading_preview = false;
                        self.log.push_str(&format!("[Ошибка] Считывание точек предпросмотра: {}\n", e));
                        preview_done = true;
                    }
                }
            }
        }
        if preview_done {
            self.preview_rx = None;
            self.preview_cancel = None;
        }
        if self.is_loading_preview {
            ctx.request_repaint();
        }

        // Обработка Drag & Drop файлов в окно программы
        let is_hovering_file = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped_files.is_empty() {
            if let Some(first) = dropped_files.first() {
                if let Some(path) = &first.path {
                    self.load_input_file(path.clone());
                }
            }
        }

        // Верхняя панель
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Выбрать входной файл").clicked() {
                    if let Some(path) = FileDialog::new()
                        .add_filter("Облака точек (*.las, *.laz, *.e57)", &["las", "laz", "e57"])
                        .add_filter("E57 (*.e57)", &["e57"])
                        .add_filter("LAS / LAZ (*.las, *.laz)", &["las", "laz"])
                        .add_filter("Все файлы (*.*)", &["*"])
                        .pick_file()
                    {
                        self.load_input_file(path);
                    }
                }

                if ui.button("Сохранить результат").clicked() {
                    let mut dialog = FileDialog::new();
                    let default_ext = if let Some(input) = &self.input_file {
                        if detect_format(input) == Some(PointCloudFormat::E57) {
                            "e57"
                        } else {
                            "las"
                        }
                    } else {
                        "las"
                    };

                    if let Some(input) = &self.input_file {
                        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
                        dialog = dialog.set_file_name(format!("{}_processed.{}", stem, default_ext));
                    }

                    if default_ext == "e57" {
                        dialog = dialog
                            .add_filter("E57 (*.e57)", &["e57"])
                            .add_filter("LAS (*.las)", &["las"]);
                    } else {
                        dialog = dialog
                            .add_filter("LAS (*.las)", &["las"])
                            .add_filter("E57 (*.e57)", &["e57"]);
                    }
                    dialog = dialog.add_filter("Все файлы (*.*)", &["*"]);

                    if let Some(path) = dialog.save_file() {
                        let new_path = add_extension_if_needed(&path, default_ext);
                        self.output_file = Some(new_path);
                    }
                }

                let can_3d = self.input_file.is_some();
                if ui.add_enabled(can_3d, egui::Button::new(" 3D Просмотр и Limit Box")).clicked() {
                    self.ensure_preview_points();
                    self.viewport_3d.show_window = true;
                }
            });
        });

        // Центральная панель
        egui::CentralPanel::default().show(ctx, |ui| {
            // Drag & Drop зона
            let frame_stroke = if is_hovering_file {
                egui::Stroke::new(2.5, egui::Color32::from_rgb(0, 220, 255))
            } else {
                egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 65, 80))
            };

            egui::Frame::group(ui.style())
                .stroke(frame_stroke)
                .inner_margin(egui::Margin::same(8.0))
                .show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        if is_hovering_file {
                            ui.colored_label(
                                egui::Color32::from_rgb(0, 220, 255),
                                " Отпустите файл для мгновенной загрузки (.las, .laz, .e57)",
                            );
                        } else if let Some(path) = &self.input_file {
                            ui.horizontal(|ui| {
                                ui.label(" Загружен файл:");
                                let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                                ui.strong(fname);
                                if let Some(ver) = &self.source_version {
                                    ui.colored_label(egui::Color32::LIGHT_GRAY, format!("({})", ver));
                                }
                            });
                        } else {
                            ui.label(" Перетащите файл облака точек сюда (.las, .laz, .e57) или нажмите «Выбрать входной файл»");
                        }
                    });
                });

            ui.add_space(4.0);

            // Блок настроек обрезки (Limit Box)
            ui.collapsing("Настройки обрезки (Limit Box)", |ui| {
                ui.checkbox(&mut self.limit_box.enabled, "Включить обрезку (Limit Box)");
                let can_3d = self.input_file.is_some();
                if ui.add_enabled(can_3d, egui::Button::new(" 3D Просмотр и интерактивная настройка")).clicked() {
                    self.ensure_preview_points();
                    self.viewport_3d.show_window = true;
                }

                if self.limit_box.enabled {
                    let (dx, dy, dz) = self.limit_box.dimensions();
                    let vol = self.limit_box.volume();
                    ui.label(format!("Размеры рамки: X = {:.2} м, Y = {:.2} м, Z = {:.2} м", dx, dy, dz));
                    ui.label(format!("Объем рамки: {:.1} куб. м", vol));
                    ui.label(format!(
                        "Диапазоны: X [{:.2}..{:.2}], Y [{:.2}..{:.2}], Z [{:.2}..{:.2}] м",
                        self.limit_box.min_x, self.limit_box.max_x,
                        self.limit_box.min_y, self.limit_box.max_y,
                        self.limit_box.min_z, self.limit_box.max_z
                    ));
                } else {
                    ui.label("Обрезка отключена (обрабатывается весь объем облака)");
                }
            });

            // Блок прореживания
            ui.collapsing("Настройки прореживания", |ui| {
                ui.checkbox(&mut self.e57_use_f32, "Координаты E57 в f32 (компактный файл)")
                    .on_hover_text(
                        "32-битные координаты (f32) уменьшают размер E57-файла ~в 2 раза.\n\
                         По умолчанию ВКЛ для структурированных E57 (локальные координаты сканера, точность ~0.01 мм).\n\
                         По умолчанию ВЫКЛ (f64) для неструктурированных E57, чтобы не терять точность на больших геодезических координатах."
                    );
                ui.checkbox(&mut self.use_thinning, "Включить прореживание");
                if self.use_thinning {
                    let is_struct_e57 = self
                        .source_version
                        .as_deref()
                        .map(|v| v.contains("структурированный"))
                        .unwrap_or(false);

                    if is_struct_e57 {
                        ui.separator();
                        ui.label("Метод прореживания для структурированного E57:");
                        ui.radio_value(
                            &mut self.thinning_method,
                            ThinningMethod::GridStride,
                            "По шагу сетки k x k (с сохранением структуры сканов и панорам)",
                        );
                        ui.radio_value(
                            &mut self.thinning_method,
                            ThinningMethod::VoxelGrid,
                            "По среднему расстоянию (3D-сетка вокселей, деструктуризация)",
                        );
                        ui.separator();
                    }

                    if is_struct_e57 && self.thinning_method == ThinningMethod::GridStride {
                        ui.horizontal(|ui| {
                            ui.label("Шаг сетки k:");
                            ui.add(egui::DragValue::new(&mut self.grid_stride_k).clamp_range(2..=10));
                        });
                        let k = self.grid_stride_k;
                        let reduction = (1.0 - 1.0 / ((k * k) as f64)) * 100.0;
                        ui.label(format!(
                            "Прореживание сетки: каждый {}-й пиксель по строкам и столбцам",
                            k
                        ));
                        ui.label(format!(
                            "Количество точек и размер уменьшатся в {} раз (-{:.1}%)",
                            k * k,
                            reduction
                        ));
                    } else {
                        ui.label("Параметры прореживания по среднему расстоянию (3D-сетка вокселей):");
                        ui.horizontal(|ui| {
                            ui.label("Среднее расстояние между точками:");
                            ui.add(
                                egui::Slider::new(&mut self.voxel_size, 0.001..=0.5)
                                    .step_by(0.001)
                                    .suffix(" м"),
                            );
                        });
                        ui.label(format!(
                            "• Текущее среднее расстояние: {:.3} м ({} мм)",
                            self.voxel_size,
                            (self.voxel_size * 1000.0).round() as i64
                        ));

                        let can_eval = self.input_file.is_some() && !self.is_processing;
                        ui.horizontal(|ui| {
                            if ui.add_enabled(can_eval, egui::Button::new("Оценка средней плотности")).clicked() {
                                if !self.sample_points.is_empty() {
                                    let count = self.sample_points.len();
                                    self.cached_eval_voxel_size = 0.0;
                                    self.density_eval_status = Some(format!(
                                        "Оценка выполнена по выборке из {} точек",
                                        format_number_with_spaces(count)
                                    ));
                                } else {
                                    self.ensure_preview_points();
                                    self.density_eval_status = Some("Запущено считывание точек облака в фоновом режиме...".to_string());
                                }
                            }
                            if let Some(ref status) = self.density_eval_status {
                                ui.label(status);
                            }
                        });

                        // Прогнозирование размера файла
                        if self.source_file_size > 0 && self.source_point_count > 0 {
                            if (self.cached_eval_voxel_size - self.voxel_size).abs() > 0.0001 {
                                let total_pts = self.source_point_count as f64;
                                let s_box = self.cloud_surface_area.unwrap_or(15000.0);
                                let d = self.voxel_size;
                                let d2 = d * d;
                                let lambda = (total_pts * d2) / s_box;

                                let ratio = if lambda > 0.000001 {
                                    (1.0 - (-lambda).exp()) / lambda
                                } else {
                                    1.0
                                };

                                let expected_kept = (total_pts * ratio).clamp(1.0, total_pts);
                                let red_pct = ((total_pts - expected_kept) / total_pts) * 100.0;

                                let in_fmt = self.input_file.as_ref().and_then(|p| detect_format(p));
                                let out_fmt = self.output_file.as_ref().and_then(|p| detect_format(p)).or(in_fmt);

                                let est_bytes = if in_fmt == Some(PointCloudFormat::Las) && out_fmt == Some(PointCloudFormat::Las) {
                                    let header_bytes = 1024u64;
                                    let point_record_len = (self.source_file_size.saturating_sub(header_bytes)) / self.source_point_count.max(1) as u64;
                                    header_bytes + (expected_kept as u64) * point_record_len.max(28)
                                } else {
                                    let overhead = (self.source_file_size as f64 * 0.05).min(50.0 * 1024.0 * 1024.0);
                                    let data_bytes = (self.source_file_size as f64) - overhead;
                                    (overhead + data_bytes * ratio) as u64
                                };

                                self.cached_eval_voxel_size = self.voxel_size;
                                self.cached_reduction_pct = red_pct;
                                self.cached_est_bytes = est_bytes;
                            }

                            ui.label(format!(
                                "• Расчетный размер файла: {} (сокращение ~{:.1}%)",
                                format_file_size(self.cached_est_bytes),
                                self.cached_reduction_pct
                            ));
                        }

                        ui.label("• Пространство разбивается на 3D-кубы (воксели), в каждом остаётся одна точка.");
                    }
                }
            });

            // Блок масштабирования
            ui.collapsing("Настройки масштабирования", |ui| {
                ui.checkbox(&mut self.use_scaling, "Включить масштабирование координат");
                if self.use_scaling {
                    ui.label("Коэффициент масштабирования (умножение координат):");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Slider::new(&mut self.scale_factor, 0.0001..=10.0)
                                .logarithmic(true)
                                .suffix("x"),
                        );
                    });
                    ui.horizontal(|ui| {
                        if ui.button("1:1000 (мм -> м)").clicked() {
                            self.scale_factor = 0.001;
                        }
                        if ui.button("1:100 (см -> м)").clicked() {
                            self.scale_factor = 0.01;
                        }
                        if ui.button("1000:1 (м -> мм)").clicked() {
                            self.scale_factor = 1000.0;
                        }
                        if ui.button("1.0 (Без изменений)").clicked() {
                            self.scale_factor = 1.0;
                        }
                    });
                    ui.label(format!("• Текущий масштаб: {:.4}x", self.scale_factor));
                }
            });

            let can_process = self.input_file.is_some()
                && self.output_file.is_some()
                && self.input_file != self.output_file
                && !self.is_processing;

            if ui.add_enabled(can_process, egui::Button::new("Начать обработку")).clicked() {
                let input = self.input_file.as_ref().unwrap().clone();
                let output = self.output_file.as_ref().unwrap().clone();

                let voxel = if self.use_thinning { self.voxel_size } else { 0.0 };
                let scale = if self.use_scaling { self.scale_factor } else { 1.0 };
                let use_thinning = self.use_thinning;
                let grid_stride_k = self.grid_stride_k;
                let thinning_method = self.thinning_method;
                let e57_use_f32 = self.e57_use_f32;
                let limit_box = self.limit_box;

                self.log.clear();
                self.result_point_count = 0;
                self.progress = 0.0;
                self.is_processing = true;

                let (tx, rx) = std::sync::mpsc::channel();
                self.worker_rx = Some(rx);

                let ctx_clone = ctx.clone();
                std::thread::spawn(move || {
                    let mut log_fn = |msg: &str| {
                        let _ = tx.send(WorkerMsg::Log(msg.to_string()));
                        ctx_clone.request_repaint();
                    };
                    let mut prog_fn = |p: f32| {
                        let _ = tx.send(WorkerMsg::Progress(p));
                        ctx_clone.request_repaint();
                    };

                    let start_time = Instant::now();
                    match process_point_cloud(
                        &input,
                        &output,
                        voxel,
                        scale,
                        use_thinning,
                        grid_stride_k,
                        thinning_method,
                        e57_use_f32,
                        &limit_box,
                        &mut log_fn,
                        &mut prog_fn,
                    ) {
                        Ok(count) => {
                            let elapsed = start_time.elapsed();
                            let _ = tx.send(WorkerMsg::Finished {
                                result_point_count: count,
                                elapsed,
                                scale,
                            });
                            ctx_clone.request_repaint();
                        }
                        Err(e) => {
                            let _ = tx.send(WorkerMsg::Error(format!("{:#}", e)));
                            ctx_clone.request_repaint();
                        }
                    }
                });
            }

            if self.is_processing {
                ui.label("Идет многопоточная обработка (Rayon)...");
                let progress = self.progress;
                ui.add(egui::ProgressBar::new(progress / 100.0).text(format!("{:.2}%", progress)));
            }

            // Путь к файлам
            if let Some(input) = &self.input_file {
                ui.label(format!("Входной файл: {}", input.display()));
            }
            if let Some(output) = &self.output_file {
                ui.label(format!("Выходной файл: {}", output.display()));
            }

            // Информация о файлах
            if let Some(version) = &self.source_version {
                ui.label(format!("Формат исходного файла: {}", version));
            } else {
                ui.label("Формат исходного файла: не определен");
            }

            if self.source_point_count > 0 {
                ui.label(format!("Всего точек/ячеек в исходном файле: {}", format_number_with_spaces(self.source_point_count)));
            }

            if self.result_point_count > 0 {
                let reduction_percent = if self.source_point_count > 0 {
                    100.0 - ((self.result_point_count as f64 / self.source_point_count as f64) * 100.0)
                } else {
                    0.0
                };
                let time_str = format_duration(self.elapsed_time);

                ui.label(format!("Активных точек в результате: {}", format_number_with_spaces(self.result_point_count)));
                if self.use_thinning {
                    ui.label(format!("Сокращение данных: {:.2}%", reduction_percent));
                }
                if self.limit_box.enabled {
                    ui.label("Применен фильтр обрезки Limit Box");
                }
                if self.use_scaling && self.scale_factor != 1.0 {
                    ui.label(format!("Масштаб: {:.3}x", self.scale_factor));
                }
                ui.label(format!("Время выполнения: {}", time_str));
            }

            // Логирование
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    for line in self.log.lines() {
                        let color = if line.starts_with("[Ошибка]") || line.starts_with("Ошибка:") {
                            egui::Color32::from_rgb(255, 100, 100)
                        } else if line.starts_with("[Конец]") || line.starts_with("Записано точек:") || line.starts_with("Обработка завершена.") {
                            egui::Color32::from_rgb(100, 255, 100)
                        } else if line.starts_with("[Старт]") || line.starts_with("Инициализирован") {
                            egui::Color32::from_rgb(100, 200, 255)
                        } else {
                            egui::Color32::LIGHT_GRAY
                        };
                        ui.colored_label(color, line);
                    }
                });
        });

        // Отдельное модальное 3D-окно для предпросмотра и настройки Limit Box
        let mut show_3d = self.viewport_3d.show_window;
        if show_3d {
            let screen_rect = ctx.screen_rect();
            let max_canvas_h = (screen_rect.height() - 90.0).max(450.0);
            let canvas_height = self.viewport_3d.canvas_height.clamp(400.0, max_canvas_h);

            egui::Window::new("3D Просмотр и обрезка (Limit Box)")
                .open(&mut show_3d)
                .default_size([1040.0, canvas_height + 40.0])
                .min_width(750.0)
                .min_height(480.0)
                .resize(|r| r.max_size(screen_rect.size()))
                .resizable(true)
                .show(ctx, |ui| {
                    // Поддержка растягивания за системную рамку окна egui
                    let avail_h = ui.available_height();
                    if ui.input(|i| i.pointer.primary_down()) && (avail_h - canvas_height).abs() > 4.0 && avail_h >= 400.0 {
                        self.viewport_3d.canvas_height = avail_h.clamp(400.0, max_canvas_h);
                    }

                    ui.horizontal_top(|ui| {
                        // Левая часть: 3D canvas + интерактивная полоса изменения высоты
                        let canvas_width = (ui.available_width() - 335.0).max(300.0);
                        let canvas_size = egui::vec2(canvas_width, canvas_height);

                        ui.vertical(|ui| {
                            ui.set_width(canvas_width);

                            let (response, painter) = ui.allocate_painter(canvas_size, egui::Sense::drag());
                            let canvas_rect = response.rect;

                        // Вращение, зум и панорамирование
                        if response.dragged_by(egui::PointerButton::Primary) {
                            let delta = response.drag_delta();
                            if ui.input(|i| i.modifiers.shift) {
                                self.viewport_3d.pan_offset += delta;
                            } else {
                                self.viewport_3d.yaw -= delta.x * 0.008;
                                self.viewport_3d.pitch = (self.viewport_3d.pitch + delta.y * 0.008).clamp(-1.54, 1.54);
                            }
                        }
                        if response.dragged_by(egui::PointerButton::Secondary) || response.dragged_by(egui::PointerButton::Middle) {
                            self.viewport_3d.pan_offset += response.drag_delta();
                        }
                        if response.hovered() {
                            let scroll = ui.input(|i| i.scroll_delta.y);
                            if scroll != 0.0 {
                                let factor = if scroll > 0.0 { 0.88 } else { 1.13 };
                                let (ex, ey, ez) = self.viewport_3d.cloud_extent;
                                let diag = ((ex * ex + ey * ey + ez * ez).sqrt() as f32).max(1.0);
                                self.viewport_3d.distance = (self.viewport_3d.distance * factor).clamp(diag * 0.002, diag * 20.0);
                            }
                        }

                        // Темно-серый фон 3D сцены
                        painter.rect_filled(canvas_rect, 0.0, egui::Color32::from_rgb(22, 24, 30));

                        if self.is_loading_preview && self.sample_points.is_empty() {
                            painter.text(
                                canvas_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                "Запуск потокового считывания точек...\nПожалуйста, подождите (0.3 сек)",
                                egui::FontId::proportional(15.0),
                                egui::Color32::from_rgb(0, 230, 255),
                            );
                        }

                        let screen_center = canvas_rect.center() + self.viewport_3d.pan_offset;
                        let focal = canvas_height * 0.9;
                        let center = self.viewport_3d.cloud_center;
                        let yaw = self.viewport_3d.yaw;
                        let pitch = self.viewport_3d.pitch;
                        let distance = self.viewport_3d.distance;
                        let min_z = self.viewport_3d.bounds_min.2;
                        let max_z = self.viewport_3d.bounds_max.2;
                        let limit_box = self.limit_box;
                        let hide_outside = self.viewport_3d.hide_outside;
                        let color_mode = self.viewport_3d.color_mode;

                        // Параллельная проекция точек облака с помощью Rayon
                        // Во время вращения/зума показываем до 250 тыс. точек (60 FPS), в покое — полный 1+ млн точек
                        let is_orbiting = response.dragged() || (response.hovered() && ui.input(|i| i.scroll_delta.y.abs() > 0.0));
                        let draw_step = if is_orbiting && self.sample_points.len() > 250_000 {
                            (self.sample_points.len() / 250_000).max(1)
                        } else {
                            1
                        };
                        let visible_points: Vec<(egui::Pos2, egui::Color32)> = self.sample_points
                            .par_iter()
                            .step_by(draw_step)
                            .filter_map(|pt| {
                                let is_inside = limit_box.contains(pt.x, pt.y, pt.z);
                                if hide_outside && !is_inside {
                                    return None;
                                }
                                let (pos, _d) = project_3d_point((pt.x, pt.y, pt.z), center, yaw, pitch, distance, screen_center, focal)?;
                                if canvas_rect.contains(pos) {
                                    let col = match color_mode {
                                        ColorMode::Rgb => {
                                            if let Some(c) = pt.color {
                                                if is_inside {
                                                    c
                                                } else {
                                                    egui::Color32::from_rgba_unmultiplied(c.r() / 3, c.g() / 3, c.b() / 3, 50)
                                                }
                                            } else {
                                                height_to_color(pt.z, min_z, max_z, is_inside)
                                            }
                                        }
                                        ColorMode::Height => height_to_color(pt.z, min_z, max_z, is_inside),
                                    };
                                    Some((pos, col))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        // Отрисовка точек через батчи egui::Mesh (высочайшая скорость GPU и защита от переполнения буфера)
                        let r = self.viewport_3d.point_size;
                        for batch in visible_points.chunks(200_000) {
                            let mut mesh = egui::Mesh::default();
                            mesh.vertices.reserve(batch.len() * 3);
                            mesh.indices.reserve(batch.len() * 3);
                            for &(pos, col) in batch {
                                let idx = mesh.vertices.len() as u32;
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(pos.x, pos.y - r), uv: egui::epaint::WHITE_UV, color: col });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(pos.x - r, pos.y + r), uv: egui::epaint::WHITE_UV, color: col });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(pos.x + r, pos.y + r), uv: egui::epaint::WHITE_UV, color: col });
                                mesh.indices.push(idx);
                                mesh.indices.push(idx + 1);
                                mesh.indices.push(idx + 2);
                            }
                            painter.add(egui::Shape::Mesh(mesh));
                        }

                        // Отрисовка каркаса Limit Box (с учётом поворота в плоскости XY)
                        let lb = &self.limit_box;
                        let corners = lb.corners();

                        let proj_corners: Vec<Option<egui::Pos2>> = corners
                            .iter()
                            .map(|&c| project_3d_point(c, center, yaw, pitch, distance, screen_center, focal).map(|(pos, _)| pos))
                            .collect();

                        let edges = [
                            (0, 1), (1, 2), (2, 3), (3, 0), // Нижняя грань
                            (4, 5), (5, 6), (6, 7), (7, 4), // Верхняя грань
                            (0, 4), (1, 5), (2, 6), (3, 7), // Стойки
                        ];

                        let edge_color = if lb.enabled {
                            egui::Color32::from_rgb(0, 240, 255)
                        } else {
                            egui::Color32::from_rgba_unmultiplied(120, 130, 150, 100)
                        };

                        for (i, j) in edges {
                            if let (Some(p1), Some(p2)) = (proj_corners[i], proj_corners[j]) {
                                painter.line_segment([p1, p2], egui::Stroke::new(2.5, edge_color));
                            }
                        }

                        // Угловые маркеры
                        for p in &proj_corners {
                            if let Some(pos) = p {
                                painter.circle_filled(*pos, 3.5, egui::Color32::from_rgb(255, 230, 0));
                            }
                        }

                        // 3D-компас (Оси X, Y, Z) в нижнем левом углу
                        let triad_center = canvas_rect.left_bottom() + egui::vec2(45.0, -45.0);
                        let axis_len = 28.0f32;
                        let (sin_y, cos_y) = yaw.sin_cos();
                        let (sin_p, cos_p) = pitch.sin_cos();

                        let project_axis = |dx: f32, dy: f32, dz: f32| -> egui::Pos2 {
                            let x_cam = dx * cos_y + dy * sin_y;
                            let y_cam = -dx * sin_y * sin_p + dy * cos_y * sin_p + dz * cos_p;
                            egui::pos2(triad_center.x + x_cam * axis_len, triad_center.y - y_cam * axis_len)
                        };

                        let p_x = project_axis(1.0, 0.0, 0.0);
                        let p_y = project_axis(0.0, 1.0, 0.0);
                        let p_z = project_axis(0.0, 0.0, 1.0);

                        painter.line_segment([triad_center, p_x], egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 60, 60)));
                        painter.text(p_x, egui::Align2::CENTER_CENTER, "X", egui::FontId::proportional(12.0), egui::Color32::from_rgb(255, 100, 100));

                        painter.line_segment([triad_center, p_y], egui::Stroke::new(2.5, egui::Color32::from_rgb(60, 255, 60)));
                        painter.text(p_y, egui::Align2::CENTER_CENTER, "Y", egui::FontId::proportional(12.0), egui::Color32::from_rgb(100, 255, 100));

                        painter.line_segment([triad_center, p_z], egui::Stroke::new(2.5, egui::Color32::from_rgb(60, 120, 255)));
                        painter.text(p_z, egui::Align2::CENTER_CENTER, "Z", egui::FontId::proportional(12.0), egui::Color32::from_rgb(120, 180, 255));

                        // Информационная плашка потоковой загрузки
                        if self.is_loading_preview && !self.sample_points.is_empty() {
                            let pill_rect = egui::Rect::from_min_size(canvas_rect.left_top() + egui::vec2(12.0, 12.0), egui::vec2(310.0, 28.0));
                            painter.rect_filled(pill_rect, 5.0, egui::Color32::from_rgba_unmultiplied(15, 20, 28, 220));
                            painter.rect_stroke(pill_rect, 5.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 200, 255)));
                            let text = format!("⚡ Потоковая загрузка: {} точек...", format_number_with_spaces(self.sample_points.len()));
                            painter.text(pill_rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(12.5), egui::Color32::from_rgb(0, 230, 255));
                        }

                        // Полоса изменения высоты окна (drag handle) под холстом
                        let (bar_rect, bar_resp) = ui.allocate_exact_size(egui::vec2(canvas_width, 8.0), egui::Sense::drag());
                        if bar_resp.dragged() {
                            self.viewport_3d.canvas_height = (self.viewport_3d.canvas_height + bar_resp.drag_delta().y).clamp(400.0, max_canvas_h);
                        }
                        if bar_resp.hovered() || bar_resp.dragged() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                            ui.painter().rect_filled(bar_rect, 2.0, egui::Color32::from_rgb(0, 210, 255));
                        } else {
                            ui.painter().rect_filled(bar_rect, 2.0, egui::Color32::from_rgb(45, 50, 60));
                        }
                    });

                    // Правая часть: панель управления Limit Box
                    ui.vertical(|ui| {
                        ui.set_width(320.0);
                        egui::ScrollArea::vertical()
                            .max_height(canvas_height + 15.0)
                            .auto_shrink([false; 2])
                            .show(ui, |ui| {
                                ui.heading("Параметры Limit Box");
                                ui.checkbox(&mut self.limit_box.enabled, "Включить обрезку (Limit Box)");

                                ui.allocate_ui_with_layout(
                                    egui::vec2(ui.available_width(), 22.0),
                                    egui::Layout::left_to_right(egui::Align::Center),
                                    |ui| {
                                        if self.is_loading_preview {
                                            ui.colored_label(egui::Color32::from_rgb(0, 220, 255), "Считывание точек...");
                                            if ui.button("⏹ Стоп (достаточно)").on_hover_text("Остановить фоновое считывание и зафиксировать текущие точки").clicked() {
                                                if let Some(ref c) = self.preview_cancel {
                                                    c.store(true, Ordering::Relaxed);
                                                }
                                                self.is_loading_preview = false;
                                            }
                                        } else {
                                            ui.colored_label(egui::Color32::from_gray(130), "Предпросмотр готов");
                                        }
                                    },
                                );

                                ui.separator();
                                ui.label("Управление камерой:");
                                ui.horizontal(|ui| {
                                    if ui.button("Изо").clicked() {
                                        self.viewport_3d.yaw = -0.785;
                                        self.viewport_3d.pitch = 0.52;
                                        self.viewport_3d.pan_offset = egui::Vec2::ZERO;
                                    }
                                    if ui.button("Сверху").clicked() {
                                        self.viewport_3d.yaw = 0.0;
                                        self.viewport_3d.pitch = 1.54;
                                        self.viewport_3d.pan_offset = egui::Vec2::ZERO;
                                    }
                                    if ui.button("Спереди").clicked() {
                                        self.viewport_3d.yaw = 0.0;
                                        self.viewport_3d.pitch = 0.0;
                                        self.viewport_3d.pan_offset = egui::Vec2::ZERO;
                                    }
                                    if ui.button("Сбоку").clicked() {
                                        self.viewport_3d.yaw = 1.5708;
                                        self.viewport_3d.pitch = 0.0;
                                        self.viewport_3d.pan_offset = egui::Vec2::ZERO;
                                    }
                                    if ui.button("Сброс").clicked() {
                                        let (dx, dy, dz) = self.viewport_3d.cloud_extent;
                                        let diag = (dx * dx + dy * dy + dz * dz).sqrt().max(10.0);
                                        self.viewport_3d.distance = (diag * 1.3) as f32;
                                        self.viewport_3d.yaw = -0.785;
                                        self.viewport_3d.pitch = 0.52;
                                        self.viewport_3d.pan_offset = egui::Vec2::ZERO;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Высота:");
                                    if ui.selectable_label((canvas_height - 500.0).abs() < 15.0, "500").clicked() {
                                        self.viewport_3d.canvas_height = 500.0;
                                    }
                                    if ui.selectable_label((canvas_height - 650.0).abs() < 15.0, "650").clicked() {
                                        self.viewport_3d.canvas_height = 650.0;
                                    }
                                    if ui.selectable_label((canvas_height - 820.0).abs() < 15.0, "820").clicked() {
                                        self.viewport_3d.canvas_height = 820.0;
                                    }
                                    if ui.button("⛶ Макс").on_hover_text("Распахнуть на всю высоту экрана").clicked() {
                                        self.viewport_3d.canvas_height = max_canvas_h;
                                    }
                                });

                                    ui.separator();
                                    ui.horizontal(|ui| {
                                        ui.label("Цвет:");
                                        ui.radio_value(&mut self.viewport_3d.color_mode, ColorMode::Rgb, "RGB");
                                        ui.radio_value(&mut self.viewport_3d.color_mode, ColorMode::Height, "Высота Z");
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Выборка:");
                                        let dv = ui.add(
                                            egui::DragValue::new(&mut self.preview_sample_percent)
                                                .speed(0.01)
                                                .clamp_range(0.001..=100.0)
                                                .max_decimals(3)
                                                .suffix(" %"),
                                        );
                                        if ui.button("Обновить").on_hover_text("Перезагрузить облако с новым % выборки").clicked()
                                            || (dv.lost_focus() && (self.cached_preview_percent - self.preview_sample_percent).abs() > 1e-6)
                                        {
                                            self.reload_preview_points();
                                        }
                                        if ui.button("×0.5").on_hover_text("Уменьшить % выборки вдвое").clicked() {
                                            self.preview_sample_percent = (self.preview_sample_percent * 0.5).clamp(0.001, 100.0);
                                            self.reload_preview_points();
                                        }
                                        if ui.button("×2").on_hover_text("Увеличить % выборки вдвое").clicked() {
                                            self.preview_sample_percent = (self.preview_sample_percent * 2.0).clamp(0.001, 100.0);
                                            self.reload_preview_points();
                                        }
                                    });

                                    ui.separator();
                                    let b_min = self.viewport_3d.bounds_min;
                                    let b_max = self.viewport_3d.bounds_max;

                                    ui.group(|ui| {
                                        ui.colored_label(egui::Color32::from_rgb(255, 120, 120), "Ось X (Восток / Длина):");
                                        ui.horizontal(|ui| {
                                            ui.label("Min:");
                                            ui.add(egui::DragValue::new(&mut self.limit_box.min_x).speed(0.2).clamp_range(b_min.0..=self.limit_box.max_x.max(b_min.0)).suffix(" м"));
                                            ui.label("Max:");
                                            ui.add(egui::DragValue::new(&mut self.limit_box.max_x).speed(0.2).clamp_range(self.limit_box.min_x.min(b_max.0)..=b_max.0).suffix(" м"));
                                        });
                                    });

                                    ui.group(|ui| {
                                        ui.colored_label(egui::Color32::from_rgb(120, 255, 120), "Ось Y (Север / Ширина):");
                                        ui.horizontal(|ui| {
                                            ui.label("Min:");
                                            ui.add(egui::DragValue::new(&mut self.limit_box.min_y).speed(0.2).clamp_range(b_min.1..=self.limit_box.max_y.max(b_min.1)).suffix(" м"));
                                            ui.label("Max:");
                                            ui.add(egui::DragValue::new(&mut self.limit_box.max_y).speed(0.2).clamp_range(self.limit_box.min_y.min(b_max.1)..=b_max.1).suffix(" м"));
                                        });
                                    });

                                    ui.group(|ui| {
                                        ui.colored_label(egui::Color32::from_rgb(120, 180, 255), "Ось Z (Высота):");
                                        ui.horizontal(|ui| {
                                            ui.label("Min:");
                                            ui.add(egui::DragValue::new(&mut self.limit_box.min_z).speed(0.2).clamp_range(b_min.2..=self.limit_box.max_z.max(b_min.2)).suffix(" м"));
                                            ui.label("Max:");
                                            ui.add(egui::DragValue::new(&mut self.limit_box.max_z).speed(0.2).clamp_range(self.limit_box.min_z.min(b_max.2)..=b_max.2).suffix(" м"));
                                        });
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Поворот Z (XY):");
                                        ui.add(egui::Slider::new(&mut self.limit_box.yaw_deg, -180.0..=180.0).step_by(0.5).suffix("°"));
                                        if ui.button("0°").on_hover_text("Сбросить поворот рамки").clicked() {
                                            self.limit_box.yaw_deg = 0.0;
                                        }
                                    });

                                    ui.separator();
                                    let (dx, dy, dz) = self.limit_box.dimensions();
                                    let vol = self.limit_box.volume();
                                    ui.label(format!("Габариты рамки: {:.2} x {:.2} x {:.2} м", dx, dy, dz));
                                    ui.label(format!("Объем рамки: {:.1} куб. м", vol));

                                    let total_sample = self.sample_points.len();
                                    let inside_sample = if self.limit_box.enabled && total_sample > 0 {
                                        self.sample_points.par_iter().filter(|p| self.limit_box.contains(p.x, p.y, p.z)).count()
                                    } else {
                                        total_sample
                                    };
                                    let pct = if total_sample > 0 {
                                        (inside_sample as f64 / total_sample as f64) * 100.0
                                    } else {
                                        100.0
                                    };
                                    ui.label(format!("Точек в выборке: {} из {} ({:.1}%)", format_number_with_spaces(inside_sample), format_number_with_spaces(total_sample), pct));

                                    ui.separator();
                                    ui.checkbox(&mut self.viewport_3d.hide_outside, "Скрывать внешние точки");
                                    ui.add(egui::Slider::new(&mut self.viewport_3d.point_size, 1.0..=5.0).text("Размер точек"));

                                    ui.add_space(8.0);
                                    if ui.button("Сбросить рамку к границам").clicked() {
                                        self.limit_box.reset_to_bounds(b_min, b_max);
                                    }
                                    if ui.button("Применить и закрыть").clicked() {
                                        self.limit_box.enabled = true;
                                        self.viewport_3d.show_window = false;
                                        if let Some(ref c) = self.preview_cancel {
                                            c.store(true, Ordering::Relaxed);
                                        }
                                        self.is_loading_preview = false;
                                    }
                                });
                        });
                    });
                });
        }
        if !show_3d && self.viewport_3d.show_window {
            if let Some(ref c) = self.preview_cancel {
                c.store(true, Ordering::Relaxed);
            }
            self.is_loading_preview = false;
        }
        self.viewport_3d.show_window = show_3d;
    }
}

// Загрузка иконки окна
fn load_icon() -> Option<eframe::IconData> {
    let icon_bytes = include_bytes!("../icon.ico");
    let image = image::load_from_memory(icon_bytes).ok()?.to_rgba8();
    let (width, height) = image.dimensions();
    Some(eframe::IconData {
        rgba: image.into_raw(),
        width,
        height,
    })
}

// Точка входа
fn main() {
    let mut options = eframe::NativeOptions::default();
    options.icon_data = load_icon();
    options.initial_window_size = Some(egui::vec2(1060.0, 860.0));
    options.min_window_size = Some(egui::vec2(680.0, 600.0));
    if let Err(error) = eframe::run_native(
        "Point Cloud Tools 2.3",
        options,
        Box::new(|_cc| Box::new(ThinLasApp::default())),
    ) {
        eprintln!("Ошибка запуска GUI: {}", error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(detect_format(Path::new("scan.las")), Some(PointCloudFormat::Las));
        assert_eq!(detect_format(Path::new("scan.laz")), Some(PointCloudFormat::Las));
        assert_eq!(detect_format(Path::new("scan.e57")), Some(PointCloudFormat::E57));
        assert_eq!(detect_format(Path::new("scan.E57")), Some(PointCloudFormat::E57));
        assert_eq!(detect_format(Path::new("scan.txt")), None);
    }

    #[test]
    fn test_add_extension_if_needed() {
        assert_eq!(
            add_extension_if_needed(Path::new("result"), "las"),
            PathBuf::from("result.las")
        );
        assert_eq!(
            add_extension_if_needed(Path::new("result.las"), "las"),
            PathBuf::from("result.las")
        );
        assert_eq!(
            add_extension_if_needed(Path::new("result.e57"), "las"),
            PathBuf::from("result.e57")
        );
    }

    #[test]
    fn test_transform_point_identity() {
        let pt = (10.0, 20.0, 30.0);
        let res = transform_point(pt, &None);
        assert_eq!(res, pt);
    }

    #[test]
    fn test_limit_box_contains_and_dimensions() {
        let mut lb = LimitBox::default();
        lb.enabled = true;
        lb.min_x = -10.0;
        lb.max_x = 10.0;
        lb.min_y = 0.0;
        lb.max_y = 20.0;
        lb.min_z = 5.0;
        lb.max_z = 15.0;

        assert!(lb.contains(0.0, 10.0, 10.0));
        assert!(lb.contains(-10.0, 0.0, 5.0));
        assert!(lb.contains(10.0, 20.0, 15.0));

        assert!(!lb.contains(-10.1, 10.0, 10.0));
        assert!(!lb.contains(0.0, 20.1, 10.0));
        assert!(!lb.contains(0.0, 10.0, 4.9));

        let (dx, dy, dz) = lb.dimensions();
        assert_eq!(dx, 20.0);
        assert_eq!(dy, 20.0);
        assert_eq!(dz, 10.0);
        assert_eq!(lb.volume(), 4000.0);

        lb.enabled = false;
        assert!(lb.contains(-999.0, 999.0, 999.0), "Disabled limit box should accept all points");
    }

    #[test]
    fn test_3d_projection() {
        let center = (0.0, 0.0, 0.0);
        let screen_center = egui::pos2(400.0, 300.0);
        let focal = 500.0;

        // Точка точно в центре
        let p_center = project_3d_point((0.0, 0.0, 0.0), center, 0.0, 0.0, 100.0, screen_center, focal);
        assert!(p_center.is_some());
        let (pos, depth) = p_center.unwrap();
        assert_eq!(pos, screen_center);
        assert_eq!(depth, 100.0);
    }

    #[test]
    fn test_format_number_with_spaces() {
        assert_eq!(format_number_with_spaces(0), "0");
        assert_eq!(format_number_with_spaces(999), "999");
        assert_eq!(format_number_with_spaces(1000), "1 000");
        assert_eq!(format_number_with_spaces(163375183), "163 375 183");
    }

    #[test]
    fn test_recommended_preview_percent() {
        assert_eq!(recommended_preview_percent(100_000, Some(PointCloudFormat::E57)), 100.0);
        assert_eq!(recommended_preview_percent(5_000_000, Some(PointCloudFormat::E57)), 10.0);
        assert_eq!(recommended_preview_percent(5_000_000, Some(PointCloudFormat::Las)), 5.0);
        assert_eq!(recommended_preview_percent(50_000_000, Some(PointCloudFormat::E57)), 1.0);
        assert!((recommended_preview_percent(306_742_831, Some(PointCloudFormat::E57)) - 0.16).abs() < 0.02);
        assert!((recommended_preview_percent(2_942_338_383, Some(PointCloudFormat::E57)) - 0.02).abs() < 0.01);
    }

    #[test]
    #[ignore]
    fn test_fast_e57_full_streaming() {
        for name in &["2222.e57", "10_27.05.2025_AG_GP.e57"] {
            let path = PathBuf::from(r"D:\IT\Dev\pc_thinning_rust\Thin_LAS_2.1\_v2.3\_point cloud").join(name);
            if !path.exists() { continue; }

            let (tx, rx) = std::sync::mpsc::channel();
            let cancel = Arc::new(AtomicBool::new(false));
            let cancel_c = cancel.clone();
            let path_c = path.clone();
            let t0 = std::time::Instant::now();
            let pct = if name.starts_with("2222") { 0.2 } else { 0.05 };
            let h = std::thread::spawn(move || {
                sample_points_streaming(&path_c, pct, cancel_c, tx)
            });

            let mut total = 0usize;
            let mut first_time = None;
            let mut min_x = f64::MAX; let mut max_x = f64::MIN;
            let mut min_y = f64::MAX; let mut max_y = f64::MIN;
            let mut min_z = f64::MAX; let mut max_z = f64::MIN;
            while let Ok(msg) = rx.recv() {
                match msg {
                    PreviewStreamMsg::Chunk(c) => {
                        if first_time.is_none() {
                            first_time = Some(t0.elapsed().as_secs_f64());
                        }
                        for p in &c {
                            min_x = min_x.min(p.x); max_x = max_x.max(p.x);
                            min_y = min_y.min(p.y); max_y = max_y.max(p.y);
                            min_z = min_z.min(p.z); max_z = max_z.max(p.z);
                        }
                        total += c.len();
                    }
                    PreviewStreamMsg::Finished => break,
                    PreviewStreamMsg::Error(e) => panic!("{}", e),
                }
            }
            h.join().unwrap().unwrap();
            println!(
                "=== {} ===\n  Total points: {}, total time: {:.3}s, first chunk: {:.3}s\n  Extent: X[{:.2}..{:.2}] ({:.2}m) Y[{:.2}..{:.2}] ({:.2}m) Z[{:.2}..{:.2}] ({:.2}m)",
                name, total, t0.elapsed().as_secs_f64(), first_time.unwrap_or(-1.0),
                min_x, max_x, max_x - min_x, min_y, max_y, max_y - min_y, min_z, max_z, max_z - min_z
            );
        }
    }

    #[test]
    #[ignore]
    fn test_fast_e57_accuracy() {
        for name in &["2222.e57", "10_27.05.2025_AG_GP.e57"] {
            let path = PathBuf::from(r"D:\IT\Dev\pc_thinning_rust\Thin_LAS_2.1\_v2.3\_point cloud").join(name);
            if !path.exists() { continue; }
            let mut reader = e57::E57Reader::from_file(&path).unwrap();
            let page_size = reader.header().page_size;
            let pcs = reader.pointclouds();
            let pc = pcs[0].clone();

            let (tx, rx) = std::sync::mpsc::channel();
            let cancel = Arc::new(AtomicBool::new(false));
            let mut paged = PagedFastReader::open(&path, page_size, 4).unwrap();
            fast_sample_e57_scan(&mut paged, &pc, 1000, 0, &cancel, &tx).unwrap().unwrap();
            drop(tx);
            let fast_pts: Vec<PreviewPoint> = rx.iter().flat_map(|m| match m {
                PreviewStreamMsg::Chunk(c) => c,
                _ => vec![],
            }).collect();

            // Verify first point of fast_pts exists verbatim as valid point #0 in E57Reader
            let mut iter = reader.pointcloud_simple(&pc).unwrap();
            iter.spherical_to_cartesian(false);
            iter.cartesian_to_spherical(false);
            iter.intensity_to_color(false);
            iter.normalize_intensity(false);
            iter.apply_pose(true);

            let first_valid = iter.find_map(|r| match r.ok()?.cartesian {
                e57::CartesianCoordinate::Valid { x, y, z } => Some((x, y, z)),
                _ => None,
            }).unwrap();
            let fp = &fast_pts[0];
            let diff = ((fp.x - first_valid.0).powi(2) + (fp.y - first_valid.1).powi(2) + (fp.z - first_valid.2).powi(2)).sqrt();
            println!("{}: first point diff = {:.9}m (got {}, valid_count={})", name, diff, fp.x, fast_pts.len());
            assert!(diff < 1e-6, "Coordinate mismatch on {}: {}", name, diff);
        }
    }

    #[test]
    #[ignore]
    fn test_e57_both_files() {
        for name in &["2222.e57", "10_27.05.2025_AG_GP.e57"] {
            let path = PathBuf::from(r"D:\IT\Dev\pc_thinning_rust\Thin_LAS_2.1\_v2.3\_point cloud").join(name);
            if !path.exists() { continue; }
            let reader = e57::E57Reader::from_file(&path).unwrap();
            let pcs = reader.pointclouds();
            let mut g_min_x = f64::MAX; let mut g_max_x = f64::MIN;
            let mut g_min_y = f64::MAX; let mut g_max_y = f64::MIN;
            let mut g_min_z = f64::MAX; let mut g_max_z = f64::MIN;

            for pc in &pcs {
                if let Some(bounds) = pc.get_cartesian_bounds() {
                    if let (Some(x0), Some(x1), Some(y0), Some(y1), Some(z0), Some(z1)) = (
                        bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max, bounds.z_min, bounds.z_max
                    ) {
                        let center_x = (x0 + x1) / 2.0;
                        let center_y = (y0 + y1) / 2.0;
                        let is_already_world = if let Some(ref t) = pc.transform {
                            let dist_to_trans = ((center_x - t.translation.x).powi(2) + (center_y - t.translation.y).powi(2)).sqrt();
                            let dist_to_zero = (center_x.powi(2) + center_y.powi(2)).sqrt();
                            dist_to_trans < dist_to_zero
                        } else {
                            true
                        };

                        let corners = [
                            (x0, y0, z0), (x0, y0, z1), (x0, y1, z0), (x0, y1, z1),
                            (x1, y0, z0), (x1, y0, z1), (x1, y1, z0), (x1, y1, z1),
                        ];
                        for (cx, cy, cz) in corners {
                            let (wx, wy, wz) = if is_already_world {
                                (cx, cy, cz)
                            } else {
                                transform_point((cx, cy, cz), &pc.transform)
                            };
                            g_min_x = g_min_x.min(wx); g_max_x = g_max_x.max(wx);
                            g_min_y = g_min_y.min(wy); g_max_y = g_max_y.max(wy);
                            g_min_z = g_min_z.min(wz); g_max_z = g_max_z.max(wz);
                        }
                    }
                }
            }
            println!("File {}:", name);
            println!("  Bounds X: [{:.2}, {:.2}], Y: [{:.2}, {:.2}], Z: [{:.2}, {:.2}]",
                g_min_x, g_max_x, g_min_y, g_max_y, g_min_z, g_max_z);
        }
    }

    #[test]
    #[ignore]
    fn test_e57_structure() {
        let path = Path::new(r"D:\IT\Dev\pc_thinning_rust\Thin_LAS_2.1\_v2.3\_point cloud\10_27.05.2025_AG_GP.e57");
        if !path.exists() {
            println!("Path not found: {:?}", path);
            return;
        }
        let start = std::time::Instant::now();
        let reader = e57::E57Reader::from_file(path).unwrap();
        println!("Header parsed in {:.3}s", start.elapsed().as_secs_f64());
        let pcs = reader.pointclouds();
        println!("Number of point clouds: {}", pcs.len());
        let mut total_rec: u64 = 0;
        for (i, pc) in pcs.iter().enumerate() {
            total_rec += pc.records;
            if i < 5 || i == pcs.len() - 1 {
                let proto_names: Vec<_> = pc.prototype.iter().map(|r| format!("{:?}", r.name)).collect();
                println!("  Scan #{}: records={}, proto={:?}, pose={:?}",
                    i, pc.records, proto_names, pc.transform.is_some());
            }
        }
        println!("Total records across all scans: {}", total_rec);

        // Now test sample_points_streaming with this file
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_clone = cancel.clone();
        let path_buf = path.to_path_buf();
        let start_stream = std::time::Instant::now();
        let h = std::thread::spawn(move || {
            sample_points_streaming(&path_buf, 0.05, cancel_clone, tx)
        });
        let mut total_pts = 0;
        let mut first_chunk_time = None;
        let mut chunks_count = 0;
        while let Ok(msg) = rx.recv() {
            match msg {
                PreviewStreamMsg::Chunk(c) => {
                    if first_chunk_time.is_none() {
                        first_chunk_time = Some(start_stream.elapsed());
                    }
                    total_pts += c.len();
                    chunks_count += 1;
                    if chunks_count <= 5 || total_pts >= 80_000 {
                        println!("  Chunk #{}: +{} pts (total: {}) at {:.3}s",
                            chunks_count, c.len(), total_pts, start_stream.elapsed().as_secs_f64());
                    }
                    if total_pts >= 5000 {
                        cancel.store(true, Ordering::Relaxed);
                        println!("  Cancelled at 5000 pts for test speed!");
                        break;
                    }
                }
                PreviewStreamMsg::Finished => {
                    println!("  Stream finished!");
                    break;
                }
                PreviewStreamMsg::Error(err) => panic!("{}", err),
            }
        }
        let _ = h.join();
        println!("Total streamed: {} in {:.3}s (first chunk: {:.3}s)",
            total_pts, start_stream.elapsed().as_secs_f64(),
            first_chunk_time.map(|t| t.as_secs_f64()).unwrap_or(-1.0));
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(500), "500 байт");
        assert_eq!(format_file_size(2048), "2 КБ");
        assert_eq!(format_file_size(10 * 1024 * 1024), "10.0 МБ");
        assert_eq!(format_file_size(4_559_309 * 1024), "4.35 ГБ");
    }

    #[test]
    fn test_density_model_prediction() {
        let n: f64 = 163_375_183.0;
        let s_box: f64 = 14037.0;
        let lambda_20: f64 = (n * 0.020 * 0.020) / s_box;
        let ratio_20 = (1.0 - (-lambda_20).exp()) / lambda_20;
        let red_20 = (1.0 - ratio_20) * 100.0;
        assert!(red_20 > 75.0 && red_20 < 82.0);
    }

    #[test]
    fn test_las_limit_box_clipping() {
        let temp_dir = std::env::temp_dir().join(format!("thin_las_2_3_lb_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let las_input = temp_dir.join("test_lb_input.las");
        let las_output = temp_dir.join("test_lb_output.las");

        {
            let mut builder = las::Builder::from((1, 2));
            builder.point_format = las::point::Format::default();
            let header = builder.into_header().unwrap();
            let mut writer = Writer::from_path(&las_input, header).unwrap();

            for i in 0..10 {
                let pt = las::Point {
                    x: i as f64 * 10.0,
                    y: 0.0,
                    z: 0.0,
                    ..Default::default()
                };
                writer.write_point(pt).unwrap();
            }
        }

        let mut dummy_log = |_msg: &str| {};
        let mut dummy_prog = |_p: f32| {};

        let mut lb = LimitBox::default();
        lb.enabled = true;
        lb.min_x = 25.0;
        lb.max_x = 65.0;
        lb.min_y = -10.0;
        lb.max_y = 10.0;
        lb.min_z = -10.0;
        lb.max_z = 10.0;

        let written = process_point_cloud(
            &las_input,
            &las_output,
            0.0,
            1.0,
            false,
            1,
            ThinningMethod::VoxelGrid,
            false,
            &lb,
            &mut dummy_log,
            &mut dummy_prog,
        ).unwrap();

        assert_eq!(written, 4, "Должно остаться ровно 4 точки внутри LimitBox!");

        let mut reader = Reader::from_path(&las_output).unwrap();
        let mut pts = Vec::new();
        while let Some(p) = reader.read_point().unwrap() {
            pts.push(p.x);
        }
        assert_eq!(pts, vec![30.0, 40.0, 50.0, 60.0]);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_las_seek() {
        let temp_dir = std::env::temp_dir().join(format!("test_seek_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let las_file = temp_dir.join("test.las");
        {
            let mut writer = Writer::from_path(&las_file, las::Builder::from((1, 2)).into_header().unwrap()).unwrap();
            for i in 0..10 {
                writer.write_point(las::Point { x: i as f64, y: 0.0, z: 0.0, ..Default::default() }).unwrap();
            }
        }
        let mut reader = Reader::from_path(&las_file).unwrap();
        reader.seek(5).unwrap();
        let pt = reader.read_point().unwrap().unwrap();
        assert_eq!(pt.x, 5.0);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
