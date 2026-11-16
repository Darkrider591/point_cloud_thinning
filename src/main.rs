#![windows_subsystem = "windows"]

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use eframe::egui;
use las::{Reader, Writer};
use roaring::RoaringBitmap;
use rfd::FileDialog;
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

// Хэширование координат в u64
fn hash_voxel<T: Hash>(t: &T) -> u64 {
    let mut s = SeaHasher::with_seeds(0, 0, 0, 0);
    t.hash(&mut s);
    s.finish()
}

// Поворот и перенос точки матрицей положения скана (Transform / Pose)
#[allow(dead_code)]
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

// Быстрая выборка точек для оценки средней плотности (без чтения всего файла)
fn sample_points_from_file(path: &Path, max_samples: usize) -> anyhow::Result<Vec<(f64, f64, f64)>> {
    let fmt = detect_format(path).unwrap_or(PointCloudFormat::Las);
    let mut points = Vec::with_capacity(max_samples);

    match fmt {
        PointCloudFormat::Las => {
            let mut reader = Reader::from_path(path)?;
            let total_pts = reader.header().number_of_points() as usize;
            let num_blocks = 5;
            let pts_per_block = (max_samples / num_blocks).max(2000);
            let skip_between = if total_pts > 500_000 {
                (total_pts / (num_blocks * 2)).clamp(10_000, 200_000)
            } else {
                0
            };

            let mut current_block_pts = 0;
            let mut blocks_read = 0;

            while let Some(pt) = reader.read_point()? {
                points.push((pt.x, pt.y, pt.z));
                current_block_pts += 1;
                if current_block_pts >= pts_per_block {
                    blocks_read += 1;
                    current_block_pts = 0;
                    if blocks_read >= num_blocks || points.len() >= max_samples {
                        break;
                    }
                    for _ in 0..skip_between {
                        if reader.read_point()?.is_none() {
                            break;
                        }
                    }
                }
            }
        }
        PointCloudFormat::E57 => {
            let mut reader = e57::E57Reader::from_file(path)?;
            let pcs = reader.pointclouds();
            let num_pcs = pcs.len().max(1);
            let per_pc = (max_samples / num_pcs).clamp(1000, max_samples);

            for pc in pcs {
                let mut iter = reader.pointcloud_simple(&pc)?;
                iter.spherical_to_cartesian(true);
                iter.cartesian_to_spherical(false);
                iter.apply_pose(true);

                let num_blocks = 5;
                let pts_per_block = (per_pc / num_blocks).max(2000);
                let skip_between = if pc.records > 500_000 {
                    ((pc.records as usize) / (num_blocks * 2)).clamp(10_000, 200_000)
                } else {
                    0
                };

                let mut current_block_pts = 0;
                let mut blocks_read = 0;

                while let Some(p) = iter.next() {
                    let p = p?;
                    if let e57::CartesianCoordinate::Valid { x, y, z } = p.cartesian {
                        points.push((x, y, z));
                        current_block_pts += 1;
                        if current_block_pts >= pts_per_block {
                            blocks_read += 1;
                            current_block_pts = 0;
                            if blocks_read >= num_blocks || points.len() >= max_samples {
                                break;
                            }
                            for _ in 0..skip_between {
                                if iter.next().is_none() {
                                    break;
                                }
                            }
                        }
                    }
                }
                if points.len() >= max_samples {
                    break;
                }
            }
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

// Вариант 1: Угловая строчно-колоночная децимация сетки E57 (k x k) с сохранением панорам
fn process_structured_e57_to_e57<L, P>(
    input_path: &Path,
    output_path: &Path,
    grid_stride_k: usize,
    scale_factor: f64,
    use_thinning: bool,
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
        "[Старт] Структурированное прореживание E57 (Вариант 1: децимация сетки k = {})",
        k
    ));
    if k > 1 {
        log(&format!(
            "Ожидаемое уменьшение точек и размера файла: в {} раз (-{:.1}%)",
            k * k,
            (1.0 - 1.0 / ((k * k) as f64)) * 100.0
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

        // Формирование прототипа
        let mut prototype = vec![
            e57::Record::CARTESIAN_X_F64,
            e57::Record::CARTESIAN_Y_F64,
            e57::Record::CARTESIAN_Z_F64,
        ];
        if has_invalid_state {
            prototype.push(e57::Record::CARTESIAN_INVALID_STATE);
        }

        // RowIndex / ColumnIndex (байт-выровненный диапазон 32-бит)
        prototype.push(e57::Record {
            name: e57::RecordName::RowIndex,
            data_type: e57::RecordDataType::Integer { min: 0, max: u32::MAX as i64 },
        });
        prototype.push(e57::Record {
            name: e57::RecordName::ColumnIndex,
            data_type: e57::RecordDataType::Integer { min: 0, max: u32::MAX as i64 },
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
        if let Some(mut transform) = scan_transform {
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

            // Вариант 1: фильтрация по сетке k x k
            if k > 1 {
                let norm_row = row - orig_row_min;
                let norm_col = col - orig_col_min;
                if norm_row % k != 0 || norm_col % k != 0 {
                    continue; // Точка отсеивается и ФИЗИЧЕСКИ НЕ пишется в файл!
                }
            }

            // Новые непрерывные индексы децимированной сетки
            let new_row = (row - orig_row_min) / k;
            let new_col = (col - orig_col_min) / k;

            let (local_x, local_y, local_z, invalid_state) = match p.cartesian {
                e57::CartesianCoordinate::Valid { x, y, z } => (x, y, z, 0i64),
                e57::CartesianCoordinate::Direction { x, y, z } => (x, y, z, 1i64),
                e57::CartesianCoordinate::Invalid => (0.0, 0.0, 0.0, 2i64),
            };

            let out_x = if scale_factor != 1.0 { local_x * scale_factor } else { local_x };
            let out_y = if scale_factor != 1.0 { local_y * scale_factor } else { local_y };
            let out_z = if scale_factor != 1.0 { local_z * scale_factor } else { local_z };

            let mut values = Vec::with_capacity(9);
            values.push(e57::RecordValue::Double(out_x));
            values.push(e57::RecordValue::Double(out_y));
            values.push(e57::RecordValue::Double(out_z));
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

// Обработка неструктурированного E57 -> E57 (потоковый классический путь)
fn process_unstructured_e57_to_e57<L, P>(
    input_path: &Path,
    output_path: &Path,
    voxel_size_m: f64,
    scale_factor: f64,
    use_thinning: bool,
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

    let file_guid = Uuid::new_v4().to_string();
    let mut e57_writer = e57::E57Writer::from_file(output_path, &file_guid)?;

    let mut prototype = vec![
        e57::Record::CARTESIAN_X_F64,
        e57::Record::CARTESIAN_Y_F64,
        e57::Record::CARTESIAN_Z_F64,
    ];
    if has_color {
        prototype.push(e57::Record::COLOR_RED_U8);
        prototype.push(e57::Record::COLOR_GREEN_U8);
        prototype.push(e57::Record::COLOR_BLUE_U8);
    }
    if has_intensity {
        prototype.push(e57::Record::INTENSITY_U16);
    }

    let pc_guid = Uuid::new_v4().to_string();
    let mut pc_writer = e57_writer.add_pointcloud(&pc_guid, prototype)?;

    let mut seen = RoaringBitmap::new();
    let mut written = 0;
    let mut processed = 0;
    let start_time = Instant::now();

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
            processed += 1;

            let (mut x, mut y, mut z) = match p.cartesian {
                e57::CartesianCoordinate::Valid { x, y, z } => (x, y, z),
                _ => continue,
            };

            if scale_factor != 1.0 {
                x *= scale_factor;
                y *= scale_factor;
                z *= scale_factor;
            }

            if use_thinning && voxel_size_m > 0.0 {
                let key = (
                    (x / voxel_size_m).round() as i64,
                    (y / voxel_size_m).round() as i64,
                    (z / voxel_size_m).round() as i64,
                );
                let hash = hash_voxel(&key);
                if seen.contains(hash as u32) {
                    continue;
                }
                seen.insert(hash as u32);
            }

            let mut record_values = Vec::with_capacity(7);
            record_values.push(e57::RecordValue::Double(x));
            record_values.push(e57::RecordValue::Double(y));
            record_values.push(e57::RecordValue::Double(z));

            if has_color {
                if let Some(c) = p.color {
                    let r = (c.red.clamp(0.0, 1.0) * 255.0).round() as i64;
                    let g = (c.green.clamp(0.0, 1.0) * 255.0).round() as i64;
                    let b = (c.blue.clamp(0.0, 1.0) * 255.0).round() as i64;
                    record_values.push(e57::RecordValue::Integer(r));
                    record_values.push(e57::RecordValue::Integer(g));
                    record_values.push(e57::RecordValue::Integer(b));
                } else {
                    record_values.push(e57::RecordValue::Integer(0));
                    record_values.push(e57::RecordValue::Integer(0));
                    record_values.push(e57::RecordValue::Integer(0));
                }
            }

            if has_intensity {
                if let Some(intensity) = p.intensity {
                    let val = (intensity.clamp(0.0, 1.0) * u16::MAX as f32).round() as i64;
                    record_values.push(e57::RecordValue::Integer(val));
                } else {
                    record_values.push(e57::RecordValue::Integer(0));
                }
            }

            pc_writer.add_point(record_values)?;
            written += 1;

            if total_points_usize > 0 && processed % (total_points_usize / 20).max(10_000) == 0 {
                let prog = (processed as f32 / total_points_usize as f32) * 90.0;
                progress(prog);
                log(&format!("Прогресс: {:.1}%", prog));
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

// Обработка LAS -> LAS
fn process_las_to_las<L, P>(
    input_path: &Path,
    output_path: &Path,
    voxel_size_m: f64,
    scale_factor: f64,
    use_thinning: bool,
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

    while let Some(mut point) = reader.read_point()? {
        processed += 1;

        if scale_factor != 1.0 {
            point.x *= scale_factor;
            point.y *= scale_factor;
            point.z *= scale_factor;
        }

        if use_thinning && voxel_size_m > 0.0 {
            let key = (
                (point.x / voxel_size_m).round() as i64,
                (point.y / voxel_size_m).round() as i64,
                (point.z / voxel_size_m).round() as i64,
            );

            let hash = hash_voxel(&key);

            if seen.contains(hash as u32) {
                continue;
            }
            seen.insert(hash as u32);
        }

        writer.write_point(point)?;
        written += 1;

        if total_points > 0 && processed % (total_points / 20).max(10_000) == 0 {
            let prog = (processed as f32 / total_points as f32) * 100.0;
            progress(prog);
            log(&format!("Прогресс: {:.1}%", prog));
        }
    }

    let duration = start_time.elapsed().as_secs_f32();
    log(&format!("Записано точек: {}", format_number_with_spaces(written)));
    log(&format!("[Конец] Обработка завершена за {:.2} сек.", duration));

    Ok(written)
}

// Обработка LAS -> E57
fn process_las_to_e57<L, P>(
    input_path: &Path,
    output_path: &Path,
    voxel_size_m: f64,
    scale_factor: f64,
    use_thinning: bool,
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

    let mut prototype = vec![
        e57::Record::CARTESIAN_X_F64,
        e57::Record::CARTESIAN_Y_F64,
        e57::Record::CARTESIAN_Z_F64,
    ];
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

    while let Some(mut point) = reader.read_point()? {
        processed += 1;

        if scale_factor != 1.0 {
            point.x *= scale_factor;
            point.y *= scale_factor;
            point.z *= scale_factor;
        }

        if use_thinning && voxel_size_m > 0.0 {
            let key = (
                (point.x / voxel_size_m).round() as i64,
                (point.y / voxel_size_m).round() as i64,
                (point.z / voxel_size_m).round() as i64,
            );
            let hash = hash_voxel(&key);
            if seen.contains(hash as u32) {
                continue;
            }
            seen.insert(hash as u32);
        }

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

        pc_writer.add_point(record_values)?;
        written += 1;

        if total_points > 0 && processed % (total_points / 20).max(10_000) == 0 {
            let p = (processed as f32 / total_points as f32) * 100.0;
            progress(p);
            log(&format!("Прогресс: {:.1}%", p));
        }
    }

    pc_writer.finalize()?;
    e57_writer.finalize()?;

    let duration = start_time.elapsed().as_secs_f32();
    log(&format!("Записано точек: {}", format_number_with_spaces(written)));
    log(&format!("[Конец] Обработка завершена за {:.2} сек.", duration));

    Ok(written)
}

// Обработка E57 -> LAS
fn process_e57_to_las<L, P>(
    input_path: &Path,
    output_path: &Path,
    voxel_size_m: f64,
    scale_factor: f64,
    use_thinning: bool,
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

    // Подготовка заголовка LAS
    let mut builder = las::Builder::from(las::Version::new(1, 4));
    builder.point_format.has_color = has_color;

    // Определение смещений из границ сканов E57
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
            processed += 1;

            let (mut x, mut y, mut z) = match p.cartesian {
                e57::CartesianCoordinate::Valid { x, y, z } => (x, y, z),
                _ => continue,
            };

            if scale_factor != 1.0 {
                x *= scale_factor;
                y *= scale_factor;
                z *= scale_factor;
            }

            if use_thinning && voxel_size_m > 0.0 {
                let key = (
                    (x / voxel_size_m).round() as i64,
                    (y / voxel_size_m).round() as i64,
                    (z / voxel_size_m).round() as i64,
                );
                let hash = hash_voxel(&key);
                if seen.contains(hash as u32) {
                    continue;
                }
                seen.insert(hash as u32);
            }

            let mut las_point = las::Point {
                x,
                y,
                z,
                ..Default::default()
            };

            if let Some(color) = p.color {
                las_point.color = Some(las::Color {
                    red: (color.red.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16,
                    green: (color.green.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16,
                    blue: (color.blue.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16,
                });
            }

            if let Some(intensity) = p.intensity {
                las_point.intensity = (intensity.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16;
            }

            writer.write_point(las_point)?;
            written += 1;

            if total_points_usize > 0 && processed % (total_points_usize / 20).max(10_000) == 0 {
                let prog = (processed as f32 / total_points_usize as f32) * 100.0;
                progress(prog);
                log(&format!("Прогресс: {:.1}%", prog));
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

    match (in_fmt, out_fmt) {
        (PointCloudFormat::Las, PointCloudFormat::Las) => {
            process_las_to_las(input_path, output_path, voxel_size_m, scale_factor, use_thinning, log, progress)
        }
        (PointCloudFormat::Las, PointCloudFormat::E57) => {
            process_las_to_e57(input_path, output_path, voxel_size_m, scale_factor, use_thinning, log, progress)
        }
        (PointCloudFormat::E57, PointCloudFormat::E57) => {
            let is_structured = if let Ok(reader) = e57::E57Reader::from_file(input_path) {
                reader.pointclouds().iter().any(|pc| pc.has_row_column())
            } else {
                false
            };

            if is_structured && thinning_method == ThinningMethod::GridStride {
                log("Инициализирован режим структурированного E57 (децимация сетки k x k + фотопанорамы)...");
                process_structured_e57_to_e57(input_path, output_path, grid_stride_k, scale_factor, use_thinning, log, progress)
            } else {
                if is_structured {
                    log("Инициализирован режим прореживания структурированного E57 по среднему расстоянию (3D-сетка вокселей)...");
                } else {
                    log("Инициализирован режим неструктурированного E57 (3D-сетка вокселей)...");
                }
                process_unstructured_e57_to_e57(input_path, output_path, voxel_size_m, scale_factor, use_thinning, log, progress)
            }
        }
        (PointCloudFormat::E57, PointCloudFormat::Las) => {
            process_e57_to_las(input_path, output_path, voxel_size_m, scale_factor, use_thinning, log, progress)
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

// Структура GUI
struct ThinLasApp {
    voxel_size: f64,
    scale_factor: f64,
    use_thinning: bool,
    use_scaling: bool,
    grid_stride_k: usize,
    thinning_method: ThinningMethod,
    input_file: Option<PathBuf>,
    output_file: Option<PathBuf>,
    source_file_size: u64,
    sample_points: Vec<(f64, f64, f64)>,
    cached_eval_voxel_size: f64,
    cached_reduction_pct: f64,
    cached_est_bytes: u64,
    cloud_surface_area: Option<f64>,
    density_eval_status: Option<String>,
    log: String,
    source_version: Option<String>,
    source_point_count: usize,
    result_point_count: usize,
    progress: f32,
    elapsed_time: Duration,
    is_processing: bool,
    worker_rx: Option<Receiver<WorkerMsg>>,
}

impl Default for ThinLasApp {
    fn default() -> Self {
        Self {
            voxel_size: 0.05,           // Шаг прореживания по умолчанию = 0.050 м или 50 мм
            scale_factor: 1.0,          // Масштаб = 1.0
            use_thinning: false,        // "Включить прореживание" = выключено по умолчанию
            use_scaling: false,         // "Включить масштабирование" = выключено по умолчанию
            grid_stride_k: 2,           // Шаг сетки k = 2 (по умолчанию в 4 раза меньше)
            thinning_method: ThinningMethod::VoxelGrid,
            input_file: None,
            output_file: None,
            source_file_size: 0,
            sample_points: Vec::new(),
            cached_eval_voxel_size: 0.0,
            cached_reduction_pct: 0.0,
            cached_est_bytes: 0,
            cloud_surface_area: None,
            density_eval_status: None,
            log: String::new(),
            source_version: None,
            source_point_count: 0,
            result_point_count: 0,
            progress: 0.0,
            elapsed_time: Duration::from_secs(0),
            is_processing: false,
            worker_rx: None,
        }
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
                        self.input_file = Some(path.clone());
                        self.source_version = None;
                        self.source_point_count = 0;
                        self.source_file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                        self.sample_points.clear();
                        self.cached_eval_voxel_size = 0.0;
                        self.cached_reduction_pct = 0.0;
                        self.cached_est_bytes = 0;
                        self.cloud_surface_area = None;
                        self.density_eval_status = None;

                        let format_hint = detect_format(&path);

                        // Проверка E57
                        if format_hint == Some(PointCloudFormat::E57) || format_hint.is_none() {
                            if let Ok(reader) = e57::E57Reader::from_file(&path) {
                                let pc_count = reader.pointclouds().len();
                                let image_count = reader.images().len();
                                let is_structured = reader.pointclouds().iter().any(|pc| pc.has_row_column());
                                let total_points: u64 =
                                    reader.pointclouds().iter().map(|pc| pc.records).sum();

                                // Извлечение габаритов (bounding box) для точного расчета физической площади поверхностей
                                let mut min_x = f64::MAX; let mut max_x = f64::MIN;
                                let mut min_y = f64::MAX; let mut max_y = f64::MIN;
                                let mut min_z = f64::MAX; let mut max_z = f64::MIN;
                                let mut has_bounds = false;
                                for pc in reader.pointclouds() {
                                    if let Some(bounds) = pc.get_cartesian_bounds() {
                                        if let (Some(x0), Some(x1), Some(y0), Some(y1), Some(z0), Some(z1)) = (
                                            bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max, bounds.z_min, bounds.z_max
                                        ) {
                                            min_x = min_x.min(x0); max_x = max_x.max(x1);
                                            min_y = min_y.min(y0); max_y = max_y.max(y1);
                                            min_z = min_z.min(z0); max_z = max_z.max(z1);
                                            has_bounds = true;
                                        }
                                    }
                                }
                                if has_bounds && max_x > min_x && max_y > min_y {
                                    let dx = max_x - min_x;
                                    let dy = max_y - min_y;
                                    let dz = (max_z - min_z).abs().max(0.1);
                                    let s_box = 2.0 * (dx * dy + dx * dz + dy * dz);
                                    self.cloud_surface_area = Some(s_box);
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

                        // Проверка LAS, если E57 не подошёл
                        if self.source_version.is_none() {
                            if let Ok(reader) = Reader::from_path(&path) {
                                let version = format!(
                                    "LAS {}.{}",
                                    reader.header().version().major,
                                    reader.header().version().minor
                                );
                                let b = reader.header().bounds();
                                let min_x = b.min.x; let max_x = b.max.x;
                                let min_y = b.min.y; let max_y = b.max.y;
                                let min_z = b.min.z; let max_z = b.max.z;
                                if max_x > min_x && max_y > min_y {
                                    let dx = max_x - min_x;
                                    let dy = max_y - min_y;
                                    let dz = (max_z - min_z).abs().max(0.1);
                                    let s_box = 2.0 * (dx * dy + dx * dz + dy * dz);
                                    self.cloud_surface_area = Some(s_box);
                                }
                                self.source_version = Some(version);
                                self.source_point_count = reader.header().number_of_points() as usize;
                                self.thinning_method = ThinningMethod::VoxelGrid;
                            }
                        }
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
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Блок прореживания
            ui.collapsing("Настройки прореживания", |ui| {
                ui.checkbox(&mut self.use_thinning, "Включить прореживание");
                if self.use_thinning {
                    let is_struct_e57 = self
                        .source_version
                        .as_deref()
                        .map(|s| s.contains("структурированный"))
                        .unwrap_or(false);

                    // Если файл не является структурированным E57, то доступен только метод VoxelGrid
                    if !is_struct_e57 {
                        self.thinning_method = ThinningMethod::VoxelGrid;
                    }

                    ui.add_space(4.0);
                    ui.group(|ui| {
                        ui.label(egui::RichText::new("Способ прореживания:").strong());

                        // Вариант 1: По коэффициенту сетки (с фотопанорамами от 1-го лица)
                        let mut method_grid = self.thinning_method == ThinningMethod::GridStride && is_struct_e57;
                        let resp_grid = ui.add_enabled(
                            is_struct_e57,
                            egui::Checkbox::new(&mut method_grid, "По коэффициенту сетки (с сохранением фотопанорам от 1-го лица)")
                        );
                        if resp_grid.clicked() {
                            self.thinning_method = ThinningMethod::GridStride;
                        }
                        resp_grid.on_hover_text(
                            "Субдискретизация 2D-сетки сканера с шагом k x k.\n\
                             • Сохраняет матричную структуру скана (строки и столбцы).\n\
                             • 100% перенос фотопанорам от 1-го лица (Bubble View / RealView для САПР).\n\
                             • Физически уменьшает размер файла и число точек в k² раз (в 4, 9, 16... раз)."
                        );

                        if !is_struct_e57 {
                            ui.label(
                                egui::RichText::new("   Недоступно для LAS и неструктурированных E57 (нет сетки сканера и фотопанорам)")
                                    .small()
                                    .italics()
                                    .color(egui::Color32::from_rgb(180, 180, 180))
                            );
                        } else {
                            ui.label(
                                egui::RichText::new("   Децимация сетки k x k: сохраняет структуру скана и панорамы Bubble View, снижает размер в k² раз")
                                    .small()
                                    .color(egui::Color32::from_rgb(120, 220, 120))
                            );
                        }

                        ui.add_space(2.0);

                        // Вариант 2: По среднему расстоянию (3D-сетка вокселей)
                        let mut method_voxel = self.thinning_method == ThinningMethod::VoxelGrid;
                        let resp_voxel = ui.add(
                            egui::Checkbox::new(&mut method_voxel, "По среднему расстоянию (3D-сетка вокселей)")
                        );
                        if resp_voxel.clicked() {
                            self.thinning_method = ThinningMethod::VoxelGrid;
                        }
                        resp_voxel.on_hover_text(
                            "Пространственное 3D-прореживание кубической сеткой вокселей.\n\
                             • Оставляет максимум 1 точку в каждом кубе заданного размера (м).\n\
                             • Обеспечивает равномерное среднее расстояние между точками по всему облаку.\n\
                             • Универсально работает для всех форматов (LAS, LAZ, E57)."
                        );
                        ui.label(
                            egui::RichText::new("   3D-сетка: равномерное среднее расстояние между точками в метрах (универсально для LAS и E57)")
                                .small()
                                .color(egui::Color32::from_rgb(160, 200, 255))
                        );
                    });

                    ui.add_space(4.0);

                    // Блок параметров выбранного метода
                    if self.thinning_method == ThinningMethod::GridStride && is_struct_e57 {
                        ui.group(|ui| {
                            ui.label(
                                egui::RichText::new(
                                    "Параметры децимации сетки сканера:",
                                )
                                .strong()
                                .color(egui::Color32::from_rgb(100, 200, 255)),
                            );

                            ui.horizontal(|ui| {
                                ui.label("Коэффициент / шаг сетки k:");
                                ui.add(
                                    egui::Slider::new(&mut self.grid_stride_k, 2..=10)
                                        .text("шаг"),
                                );
                            });

                            let k = self.grid_stride_k;
                            let factor = k * k;
                            let reduction = (1.0 - 1.0 / (factor as f64)) * 100.0;
                            ui.label(format!("• Прореживание: каждая {}-я строка и {}-я колонка матрицы скана", k, k));
                            ui.label(
                                egui::RichText::new(format!(
                                    "• Точек и размер файла: в {} раз меньше (-{:.1}%)",
                                    factor, reduction
                                ))
                                .strong()
                                .color(egui::Color32::from_rgb(120, 255, 120)),
                            );

                            if self.source_file_size > 0 {
                                let est_bytes = ((self.source_file_size as f64) / (factor as f64)).round() as u64;
                                ui.label(
                                    egui::RichText::new(format!(
                                        "• Рассчетный размер файла: {} (сокращение на {:.1}%)",
                                        format_file_size(est_bytes),
                                        reduction
                                    ))
                                    .strong()
                                    .color(egui::Color32::from_rgb(100, 220, 255)),
                                );
                            }

                            ui.label("• Топология 2D-сетки и фотопанорамы от 1-го лица (RealView) полностью сохраняются.");
                        });
                    } else {
                        ui.group(|ui| {
                            ui.label(
                                egui::RichText::new(
                                    "Параметры прореживания по среднему расстоянию (3D-сетка вокселей):",
                                )
                                .strong()
                                .color(egui::Color32::from_rgb(100, 200, 255)),
                            );

                            ui.horizontal(|ui| {
                                ui.label("Среднее расстояние между точками:");
                                ui.add(
                                    egui::Slider::new(&mut self.voxel_size, 0.001..=10.0)
                                        .text("м")
                                        .step_by(0.001)
                                        .max_decimals(3)
                                        .logarithmic(true),
                                );
                            });
                            let mm = (self.voxel_size * 1000.0).round() as i64;
                            ui.label(format!("• Текущее среднее расстояние: {:.3} м ({} мм)", self.voxel_size, mm));

                            ui.add_space(4.0);

                            // Кнопка оценки средней плотности
                            let can_eval = self.input_file.is_some() && !self.is_processing;
                            ui.horizontal(|ui| {
                                if ui.add_enabled(can_eval, egui::Button::new("Оценка средней плотности")).clicked() {
                                    if let Some(ref path) = self.input_file {
                                        match sample_points_from_file(path, 100_000) {
                                            Ok(pts) => {
                                                let count = pts.len();
                                                if self.cloud_surface_area.is_none() && !pts.is_empty() {
                                                    let mut min_x = f64::MAX; let mut max_x = f64::MIN;
                                                    let mut min_y = f64::MAX; let mut max_y = f64::MIN;
                                                    let mut min_z = f64::MAX; let mut max_z = f64::MIN;
                                                    for &(x, y, z) in &pts {
                                                        min_x = min_x.min(x); max_x = max_x.max(x);
                                                        min_y = min_y.min(y); max_y = max_y.max(y);
                                                        min_z = min_z.min(z); max_z = max_z.max(z);
                                                    }
                                                    if max_x > min_x && max_y > min_y {
                                                        let dx = max_x - min_x;
                                                        let dy = max_y - min_y;
                                                        let dz = (max_z - min_z).abs().max(0.1);
                                                        self.cloud_surface_area = Some(2.0 * (dx * dy + dx * dz + dy * dz));
                                                    }
                                                }
                                                self.sample_points = pts;
                                                self.cached_eval_voxel_size = 0.0;
                                                let density_msg = if let Some(s) = self.cloud_surface_area {
                                                    let n = self.source_point_count as f64;
                                                    let avg_dens = n / s;
                                                    let avg_dist_mm = (1.0 / avg_dens.sqrt()) * 1000.0;
                                                    format!(
                                                        "Средняя плотность: ~{:.0} точ./м² (средний шаг ~{:.1} мм), выборка {}",
                                                        avg_dens, avg_dist_mm, format_number_with_spaces(count)
                                                    )
                                                } else {
                                                    format!(
                                                        "Оценка выполнена по выборке из {} точек",
                                                        format_number_with_spaces(count)
                                                    )
                                                };
                                                self.density_eval_status = Some(density_msg);
                                            }
                                            Err(e) => {
                                                self.density_eval_status = Some(format!("Ошибка оценки: {}", e));
                                            }
                                        }
                                    }
                                }

                                if let Some(ref status) = self.density_eval_status {
                                    ui.label(egui::RichText::new(status).small().color(egui::Color32::from_rgb(180, 200, 220)));
                                }
                            });

                            // Расчетный размер файла на основе физической модели поверхности и вокселей
                            if !self.sample_points.is_empty() || self.cloud_surface_area.is_some() {
                                if (self.voxel_size - self.cached_eval_voxel_size).abs() > 1e-6 || self.cached_eval_voxel_size == 0.0 {
                                    let ratio = if let Some(s) = self.cloud_surface_area {
                                        if s > 0.0 && self.source_point_count > 0 {
                                            let n = self.source_point_count as f64;
                                            let lambda = (n * self.voxel_size * self.voxel_size) / s;
                                            if lambda > 1e-6 {
                                                ((1.0 - (-lambda).exp()) / lambda).clamp(0.0001, 1.0)
                                            } else {
                                                1.0
                                            }
                                        } else {
                                            1.0
                                        }
                                    } else {
                                        let mut seen = RoaringBitmap::new();
                                        let mut kept = 0;
                                        for &(x, y, z) in &self.sample_points {
                                            let key = (
                                                (x / self.voxel_size).round() as i64,
                                                (y / self.voxel_size).round() as i64,
                                                (z / self.voxel_size).round() as i64,
                                            );
                                            let hash = hash_voxel(&key);
                                            if !seen.contains(hash as u32) {
                                                seen.insert(hash as u32);
                                                kept += 1;
                                            }
                                        }
                                        kept as f64 / self.sample_points.len().max(1) as f64
                                    };

                                    self.cached_reduction_pct = (1.0 - ratio) * 100.0;
                                    self.cached_est_bytes = if self.source_file_size > 0 {
                                        ((self.source_file_size as f64) * ratio).round() as u64
                                    } else {
                                        0
                                    };
                                    self.cached_eval_voxel_size = self.voxel_size;
                                }

                                let reduction_pct = self.cached_reduction_pct;
                                let est_bytes = self.cached_est_bytes;

                                if reduction_pct < 2.0 {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "• Рассчетный размер файла: {} (сокращение ~{:.1}% — шаг слишком мал, размер практически не уменьшится)",
                                            format_file_size(est_bytes),
                                            reduction_pct
                                        ))
                                        .strong()
                                        .color(egui::Color32::from_rgb(255, 180, 100)),
                                    );
                                } else {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "• Рассчетный размер файла: {} (сокращение ~{:.1}%)",
                                            format_file_size(est_bytes),
                                            reduction_pct
                                        ))
                                        .strong()
                                        .color(egui::Color32::from_rgb(100, 220, 255)),
                                    );
                                }
                            } else {
                                ui.label(
                                    egui::RichText::new("• Рассчетный размер файла: нажмите кнопку «Оценка средней плотности»")
                                        .italics()
                                        .color(egui::Color32::from_rgb(160, 160, 160)),
                                );
                            }

                            ui.label("• Пространство разбивается на 3D-кубы (воксели), в каждом остаётся одна точка.");
                        });
                    }
                }
            });

            // Блок масштабирования
            ui.collapsing("Настройки масштабирования", |ui| {
                ui.checkbox(&mut self.use_scaling, "Включить масштабирование");
                if self.use_scaling {
                    ui.add(
                        egui::Slider::new(&mut self.scale_factor, 0.001..=1000.0)
                            .text("Коэффициент масштаба")
                            .logarithmic(true),
                    );
                    ui.label(format!("Текущий коэффициент: {:.3}", self.scale_factor));
                    ui.label("• < 1.0 — уменьшение облака");
                    ui.label("• = 1.0 — без изменений");
                    ui.label("• > 1.0 — увеличение облака");
                }
            });

            // Кнопка обработки
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
                ui.label("Идет обработка...");
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
                if self.use_scaling && self.scale_factor != 1.0 {
                    ui.label(format!("Масштаб: {:.3}x", self.scale_factor));
                }
                ui.label(format!("Время выполнения: {}", time_str));
            }

            // Цветное логирование
            egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                for line in self.log.lines() {
                    if line.contains("Ошибка") {
                        ui.colored_label(egui::Color32::RED, line);
                    } else if line.contains("завершена") || line.contains("Записано") || line.contains("записано") {
                        ui.colored_label(egui::Color32::GREEN, line);
                    } else if line.contains("Масштаб") {
                        ui.colored_label(egui::Color32::BLUE, line);
                    } else {
                        ui.label(line);
                    }
                }
            });
        });
    }
}

fn load_icon() -> Option<eframe::IconData> {
    let icon_bytes = include_bytes!("../icon.ico");
    let img = image::load_from_memory(icon_bytes).ok()?.into_rgba8();
    let (width, height) = img.dimensions();
    let rgba = img.into_raw();
    Some(eframe::IconData {
        rgba,
        width,
        height,
    })
}

// Точка входа
fn main() {
    let mut options = eframe::NativeOptions::default();
    options.icon_data = load_icon();
    if let Err(error) = eframe::run_native(
        "Thin_LAS_2.2",
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
        assert_eq!(detect_format(Path::new("SCAN.E57")), Some(PointCloudFormat::E57));
        assert_eq!(detect_format(Path::new("scan.xyz")), None);
    }

    #[test]
    fn test_add_extension() {
        assert_eq!(
            add_extension_if_needed(Path::new("myfile"), "las"),
            PathBuf::from("myfile.las")
        );
        assert_eq!(
            add_extension_if_needed(Path::new("myfile"), "e57"),
            PathBuf::from("myfile.e57")
        );
        assert_eq!(
            add_extension_if_needed(Path::new("myfile.las"), "e57"),
            PathBuf::from("myfile.las")
        );
        assert_eq!(
            add_extension_if_needed(Path::new("myfile.e57"), "las"),
            PathBuf::from("myfile.e57")
        );
    }

    #[test]
    fn test_transform_point_identity() {
        let pt = (10.0, 20.0, 30.0);
        let res = transform_point(pt, &None);
        assert_eq!(res, pt);
    }

    #[test]
    fn test_structured_e57_with_spherical_panorama() {
        let temp_dir = std::env::temp_dir().join(format!("thin_las_2_1_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let e57_input = temp_dir.join("structured_input.e57");
        let e57_output = temp_dir.join("structured_output.e57");

        let input_pc_guid = Uuid::new_v4().to_string();
        let input_img_guid = Uuid::new_v4().to_string();

        // 1. Создаем исходный структурированный E57 файл с сеткой 4 строки x 4 столбца (всего 16 точек)
        // и сферической фотопанорамой
        {
            let mut writer = e57::E57Writer::from_file(&e57_input, &Uuid::new_v4().to_string()).unwrap();

            let prototype = vec![
                e57::Record::CARTESIAN_X_F64,
                e57::Record::CARTESIAN_Y_F64,
                e57::Record::CARTESIAN_Z_F64,
                e57::Record::CARTESIAN_INVALID_STATE,
                e57::Record {
                    name: e57::RecordName::RowIndex,
                    data_type: e57::RecordDataType::Integer { min: 0, max: 255 },
                },
                e57::Record {
                    name: e57::RecordName::ColumnIndex,
                    data_type: e57::RecordDataType::Integer { min: 0, max: 255 },
                },
                e57::Record::COLOR_RED_U8,
                e57::Record::COLOR_GREEN_U8,
                e57::Record::COLOR_BLUE_U8,
            ];

            let mut pc_writer = writer.add_pointcloud(&input_pc_guid, prototype).unwrap();
            pc_writer.set_name(Some("Test Structured Scan 4x4".to_string()));

            for r in 0..4 {
                for c in 0..4 {
                    let is_valid = !(r == 3 && c == 3);
                    let invalid_code = if is_valid { 0 } else { 1 };
                    pc_writer.add_point(vec![
                        e57::RecordValue::Double(r as f64 * 2.0),
                        e57::RecordValue::Double(c as f64 * 2.0),
                        e57::RecordValue::Double(1.0),
                        e57::RecordValue::Integer(invalid_code),
                        e57::RecordValue::Integer(r),
                        e57::RecordValue::Integer(c),
                        e57::RecordValue::Integer(255),
                        e57::RecordValue::Integer(0),
                        e57::RecordValue::Integer(0),
                    ]).unwrap();
                }
            }

            pc_writer.finalize().unwrap();

            // Добавляем сферическую панораму
            let mut img_writer = writer.add_image(&input_img_guid).unwrap();
            img_writer.set_name("360 Panorama");
            img_writer.set_pointcloud_guid(&input_pc_guid);

            let dummy_jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46];
            let mut cursor = std::io::Cursor::new(dummy_jpeg_data);
            let props = e57::SphericalImageProperties {
                width: 1024,
                height: 512,
                pixel_width: std::f64::consts::PI / 512.0,
                pixel_height: std::f64::consts::PI / 512.0,
            };
            img_writer.add_spherical(e57::ImageFormat::Jpeg, &mut cursor, props, None).unwrap();
            img_writer.finalize().unwrap();

            writer.finalize().unwrap();
        }

        let mut dummy_log = |_msg: &str| {};
        let mut dummy_prog = |_p: f32| {};

        // 2. Выполняем децимацию сетки с шагом k = 2 (из сетки 4x4 должна получиться сетка 2x2 = 4 точки)
        let written = process_point_cloud(
            &e57_input,
            &e57_output,
            0.1,
            1.0,
            true,
            2, // k = 2
            ThinningMethod::GridStride,
            &mut dummy_log,
            &mut dummy_prog,
        ).unwrap();

        // Физически записано ровно 4 точки (в 4 раза меньше, чем 16)
        assert_eq!(written, 4);

        // 3. Проверяем результирующий E57 файл:
        {
            let mut reader = e57::E57Reader::from_file(&e57_output).unwrap();
            
            // Проверяем облако точек:
            assert_eq!(reader.pointclouds().len(), 1);
            let out_pc = &reader.pointclouds()[0];
            // ОБЯЗАТЕЛЬНО: в файле физически записано ровно 4 точки (а не 16!)
            assert_eq!(out_pc.records, 4);
            assert!(out_pc.has_row_column(), "Скан обязан остаться структурированным!");

            let mut iter = reader.pointcloud_simple(out_pc).unwrap();
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            while let Some(pt) = iter.next() {
                let pt = pt.unwrap();
                rows.push(pt.row);
                cols.push(pt.column);
            }
            assert_eq!(rows.len(), 4, "Должно быть ровно 4 точки!");
            // Новые индексы строк и колонок должны быть 0 и 1:
            assert!(rows.iter().all(|&r| r == 0 || r == 1));
            assert!(cols.iter().all(|&c| c == 0 || c == 1));

            // Проверяем фотопанораму:
            assert_eq!(reader.images().len(), 1, "Фотопанорама обязана сохраниться!");
            let out_img = &reader.images()[0];
            assert_eq!(out_img.name.as_deref(), Some("360 Panorama"));
            assert_eq!(out_img.pointcloud_guid.as_deref(), out_pc.guid.as_deref(), "Панорама должна ссылаться на новый GUID облака!");

            if let Some(e57::Projection::Spherical(spherical)) = &out_img.projection {
                assert_eq!(spherical.properties.width, 1024);
                assert_eq!(spherical.properties.height, 512);
                let mut blob_data = Vec::new();
                reader.blob(&spherical.blob.data, &mut blob_data).unwrap();
                assert_eq!(blob_data, vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46]);
            } else {
                panic!("Expected spherical projection");
            }
        }

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_structured_e57_voxel_grid_thinning() {
        let temp_dir = std::env::temp_dir().join(format!("thin_las_2_1_voxel_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let e57_input = temp_dir.join("structured_input.e57");
        let e57_output = temp_dir.join("structured_output_voxel.e57");

        let input_pc_guid = Uuid::new_v4().to_string();

        {
            let mut writer = e57::E57Writer::from_file(&e57_input, &Uuid::new_v4().to_string()).unwrap();
            let prototype = vec![
                e57::Record::CARTESIAN_X_F64,
                e57::Record::CARTESIAN_Y_F64,
                e57::Record::CARTESIAN_Z_F64,
                e57::Record {
                    name: e57::RecordName::RowIndex,
                    data_type: e57::RecordDataType::Integer { min: 0, max: 255 },
                },
                e57::Record {
                    name: e57::RecordName::ColumnIndex,
                    data_type: e57::RecordDataType::Integer { min: 0, max: 255 },
                },
            ];

            let mut pc_writer = writer.add_pointcloud(&input_pc_guid, prototype).unwrap();
            for r in 0..4 {
                for c in 0..4 {
                    pc_writer.add_point(vec![
                        e57::RecordValue::Double(r as f64 * 0.1),
                        e57::RecordValue::Double(c as f64 * 0.1),
                        e57::RecordValue::Double(1.0),
                        e57::RecordValue::Integer(r),
                        e57::RecordValue::Integer(c),
                    ]).unwrap();
                }
            }
            pc_writer.finalize().unwrap();
            writer.finalize().unwrap();
        }

        let mut dummy_log = |_msg: &str| {};
        let mut dummy_prog = |_p: f32| {};

        // Прореживаем вокселем 1.0 м (все 16 точек укладываются в диапазон 0..0.3 м, поэтому останется 1 точка)
        let written = process_point_cloud(
            &e57_input,
            &e57_output,
            1.0, // 1 метр
            1.0,
            true,
            2,
            ThinningMethod::VoxelGrid,
            &mut dummy_log,
            &mut dummy_prog,
        ).unwrap();

        assert_eq!(written, 1);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_format_number_with_spaces() {
        assert_eq!(format_number_with_spaces(0), "0");
        assert_eq!(format_number_with_spaces(999), "999");
        assert_eq!(format_number_with_spaces(1000), "1 000");
        assert_eq!(format_number_with_spaces(163375183), "163 375 183");
        assert_eq!(format_number_with_spaces(160305070), "160 305 070");
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
        // Test 20mm (voxel = 0.020)
        let lambda_20: f64 = (n * 0.020 * 0.020) / s_box;
        let ratio_20 = (1.0 - (-lambda_20).exp()) / lambda_20;
        let red_20 = (1.0 - ratio_20) * 100.0;
        assert!(red_20 > 75.0 && red_20 < 82.0, "Expected ~79% reduction at 20mm, got {:.1}%", red_20);

        // Test 8mm (voxel = 0.008)
        let lambda_8: f64 = (n * 0.008 * 0.008) / s_box;
        let ratio_8 = (1.0 - (-lambda_8).exp()) / lambda_8;
        let red_8 = (1.0 - ratio_8) * 100.0;
        assert!(red_8 > 25.0 && red_8 < 35.0, "Expected ~29% reduction at 8mm, got {:.1}%", red_8);
    }
}
