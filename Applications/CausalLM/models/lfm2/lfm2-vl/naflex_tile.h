// SPDX-License-Identifier: Apache-2.0
/**
 * @file   naflex_tile.h
 * @date   09 June 2026
 * @brief  NaFlex image tiling utilities matching HF Lfm2VlImageProcessor.
 * @author SeungBaek Hong <baek2sm@naver.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CAUSALLM_NAFLEX_TILE_H__
#define __CAUSALLM_NAFLEX_TILE_H__

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../image_util.h"

namespace causallm {

struct NaFlexTilingParams {
  int tile_size = 512;
  int max_tiles = 10;
  int min_tiles = 2;
  bool use_thumbnail = false;
  int downsample_factor = 2;
  int max_image_tokens = 256;
  int encoder_patch_size = 16;
  float max_pixels_tolerance = 2.0f;
  int min_image_tokens = 64;
};

struct TileGrid {
  int grid_w;
  int grid_h;
  int target_w;
  int target_h;
  int total_tiles;
};

struct NaFlexTileResult {
  std::vector<std::vector<float>> tiles;
  int grid_w;
  int grid_h;
  int n_tiles;
  int tile_pixel_h;
  int tile_pixel_w;
};

inline int roundByFactor(float number, int factor) {
  return static_cast<int>(std::round(number / factor)) * factor;
}

inline std::pair<int, int> findClosestAspectRatio(
  float aspect_ratio, const std::vector<std::pair<int, int>> &target_ratios,
  int orig_w, int orig_h, int image_size) {
  float best_diff = 1e30f;
  std::pair<int, int> best = {1, 1};
  long long area = static_cast<long long>(orig_w) * orig_h;
  for (const auto &r : target_ratios) {
    float ratio = static_cast<float>(r.first) / r.second;
    float diff = std::abs(aspect_ratio - ratio);
    if (diff < best_diff) {
      best_diff = diff;
      best = r;
    } else if (diff == best_diff) {
      long long target_area =
        static_cast<long long>(image_size) * image_size * r.first * r.second;
      if (area > target_area / 2)
        best = r;
    }
  }
  return best;
}

inline std::vector<std::pair<int, int>> getTargetRatios(int min_tiles,
                                                        int max_tiles) {
  std::set<std::pair<int, int>> ratio_set;
  for (int n = min_tiles; n <= max_tiles; ++n)
    for (int w = 1; w <= n; ++w)
      for (int h = 1; h <= n; ++h)
        if (w * h >= min_tiles && w * h <= max_tiles)
          ratio_set.insert({w, h});
  std::vector<std::pair<int, int>> ratios(ratio_set.begin(), ratio_set.end());
  std::stable_sort(ratios.begin(), ratios.end(), [](const auto &a,
                                                    const auto &b) {
    return a.first * a.second < b.first * b.second;
  });
  return ratios;
}

inline std::pair<int, int> getImageSizeForMaxPatches(
  int image_h, int image_w, int patch_size, int max_num_patches,
  float eps = 1e-5f) {
  auto scaled = [&](float scale, int size) -> int {
    int s =
      static_cast<int>(std::ceil(size * scale / patch_size)) * patch_size;
    return std::max(patch_size, s);
  };
  float lo = eps / 10.0f, hi = 100.0f;
  while ((hi - lo) >= eps) {
    float mid = (lo + hi) / 2.0f;
    int th = scaled(mid, image_h);
    int tw = scaled(mid, image_w);
    float n_patches = static_cast<float>(th / patch_size) * (tw / patch_size);
    if (n_patches <= max_num_patches)
      lo = mid;
    else
      hi = mid;
  }
  return {scaled(lo, image_h), scaled(lo, image_w)};
}

inline std::pair<int, int> smartResize(int height, int width,
                                       int downsample_factor,
                                       int min_image_tokens,
                                       int max_image_tokens,
                                       int encoder_patch_size) {
  int total_factor = encoder_patch_size * downsample_factor;
  long long smart_min = static_cast<long long>(min_image_tokens) *
                        encoder_patch_size * encoder_patch_size *
                        downsample_factor * downsample_factor;
  long long smart_max = static_cast<long long>(max_image_tokens) *
                        encoder_patch_size * encoder_patch_size *
                        downsample_factor * downsample_factor;

  int h_bar =
    std::max(total_factor, roundByFactor(static_cast<float>(height), total_factor));
  int w_bar =
    std::max(total_factor, roundByFactor(static_cast<float>(width), total_factor));

  long long hw = static_cast<long long>(h_bar) * w_bar;
  if (hw > smart_max) {
    float beta = std::sqrt(static_cast<float>(height * width) / smart_max);
    h_bar = std::max(total_factor,
                     static_cast<int>(std::floor(height / beta / total_factor)) *
                       total_factor);
    w_bar = std::max(total_factor,
                     static_cast<int>(std::floor(width / beta / total_factor)) *
                       total_factor);
  } else if (hw < smart_min) {
    float beta = std::sqrt(static_cast<float>(smart_min) / (height * width));
    h_bar =
      static_cast<int>(std::ceil(height * beta / total_factor)) * total_factor;
    w_bar =
      static_cast<int>(std::ceil(width * beta / total_factor)) * total_factor;
  }
  return {w_bar, h_bar};
}

inline bool isImageTooLarge(int height, int width, int max_image_tokens,
                            int encoder_patch_size, int downsample_factor,
                            float max_pixels_tolerance) {
  int total_factor = encoder_patch_size * downsample_factor;
  int h_bar = std::max(encoder_patch_size,
                       roundByFactor(static_cast<float>(height), total_factor));
  int w_bar = std::max(encoder_patch_size,
                       roundByFactor(static_cast<float>(width), total_factor));
  long long limit = static_cast<long long>(max_image_tokens) *
                    encoder_patch_size * encoder_patch_size *
                    downsample_factor * downsample_factor *
                    static_cast<long long>(max_pixels_tolerance);
  return static_cast<long long>(h_bar) * w_bar > limit;
}

inline std::vector<float>
normalizeTileChw(const unsigned char *tile_hwc, int tile_size, int ch) {
  std::vector<float> tile_chw(static_cast<size_t>(ch) * tile_size * tile_size);
  for (int c = 0; c < ch; ++c)
    for (int y = 0; y < tile_size; ++y)
      for (int x = 0; x < tile_size; ++x)
        tile_chw[c * tile_size * tile_size + y * tile_size + x] =
          (tile_hwc[(y * tile_size + x) * ch + c] / 255.0f - 0.5f) / 0.5f;
  return tile_chw;
}

inline NaFlexTileResult naflexTileImage(const std::string &image_path,
                                        const NaFlexTilingParams &params) {
  int src_w, src_h, src_ch;
  unsigned char *raw =
    stbi_load(image_path.c_str(), &src_w, &src_h, &src_ch, STBI_rgb);
  if (!raw)
    throw std::runtime_error("naflexTileImage: cannot load: " + image_path);
  int ch = 3;

  NaFlexTileResult result;
  result.tile_pixel_h = params.tile_size;
  result.tile_pixel_w = params.tile_size;

  auto [sr_w, sr_h] =
    smartResize(src_h, src_w, params.downsample_factor,
                params.min_image_tokens, params.max_image_tokens,
                params.encoder_patch_size);

  bool is_large =
    params.max_tiles > 1 &&
    isImageTooLarge(src_h, src_w, params.max_image_tokens,
                    params.encoder_patch_size, params.downsample_factor,
                    params.max_pixels_tolerance);

  if (is_large) {
    auto target_ratios = getTargetRatios(params.min_tiles, params.max_tiles);
    float aspect = static_cast<float>(src_w) / src_h;
    auto [grid_w, grid_h] =
      findClosestAspectRatio(aspect, target_ratios, src_w, src_h,
                             params.tile_size);
    int target_w = grid_w * params.tile_size;
    int target_h = grid_h * params.tile_size;

    result.grid_w = grid_w;
    result.grid_h = grid_h;

    auto resized = resizeImage(raw, src_w, src_h, ch, target_w, target_h);
    stbi_image_free(raw);
    raw = nullptr;

    for (int ty = 0; ty < grid_h; ++ty) {
      for (int tx = 0; tx < grid_w; ++tx) {
        std::vector<unsigned char> tile_hwc(
          static_cast<size_t>(params.tile_size) * params.tile_size * ch);
        for (int y = 0; y < params.tile_size; ++y) {
          int src_y = ty * params.tile_size + y;
          for (int x = 0; x < params.tile_size; ++x) {
            int src_x = tx * params.tile_size + x;
            for (int c = 0; c < ch; ++c)
              tile_hwc[(y * params.tile_size + x) * ch + c] =
                resized[(src_y * target_w + src_x) * ch + c];
          }
        }
        result.tiles.push_back(
          normalizeTileChw(tile_hwc.data(), params.tile_size, ch));
      }
    }

    if (params.use_thumbnail && grid_w * grid_h > 1) {
      auto thumb_resized =
        resizeImage(resized.data(), target_w, target_h, ch, sr_w, sr_h);
      auto thumb_tile = resizeImage(thumb_resized.data(), sr_w, sr_h, ch,
                                    params.tile_size, params.tile_size);
      result.tiles.push_back(
        normalizeTileChw(thumb_tile.data(), params.tile_size, ch));
    }
  } else {
    stbi_image_free(raw);
    raw = nullptr;
    auto tile_pixels = loadAndPreprocessImage(
      image_path, params.tile_size, params.tile_size, true);
    result.grid_w = 1;
    result.grid_h = 1;
    result.tiles.push_back(std::move(tile_pixels));
  }

  result.n_tiles = static_cast<int>(result.tiles.size());
  return result;
}

} // namespace causallm

#endif /* __CAUSALLM_NAFLEX_TILE_H__ */
