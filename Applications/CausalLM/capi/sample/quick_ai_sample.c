// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   quick_ai_sample.c
 * @date   28 May 2026
 * @brief  Minimal host sample for the quick.ai C ABI DLL
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "quick_ai.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef quick_ai_status (*quick_ai_ctx_create_fn)(const char *,
                                                  const quick_ai_init_options *,
                                                  quick_ai_ctx **);
typedef void (*quick_ai_ctx_free_fn)(quick_ai_ctx *);
typedef quick_ai_status (*quick_ai_generate_fn)(quick_ai_ctx *, const char *,
                                                quick_ai_token_cb, void *,
                                                quick_ai_usage *);
typedef const char *(*quick_ai_last_error_fn)(quick_ai_ctx *);
typedef const char *(*quick_ai_version_fn)(void);

static int print_token(const char *utf8_delta, void *user_data) {
  (void)user_data;
  if (utf8_delta != NULL) {
    fputs(utf8_delta, stdout);
    fflush(stdout);
  }
  return 0;
}

static char *json_escape(const char *text) {
  static const char hex[] = "0123456789abcdef";
  size_t len;
  size_t cap;
  size_t i;
  size_t j;
  char *out;

  if (text == NULL)
    text = "";

  len = strlen(text);
  cap = len * 6 + 1;
  out = (char *)malloc(cap);
  if (out == NULL)
    return NULL;

  j = 0;
  for (i = 0; i < len; ++i) {
    unsigned char ch;
    ch = (unsigned char)text[i];
    switch (ch) {
    case '\\':
      out[j++] = '\\';
      out[j++] = '\\';
      break;
    case '"':
      out[j++] = '\\';
      out[j++] = '"';
      break;
    case '\n':
      out[j++] = '\\';
      out[j++] = 'n';
      break;
    case '\r':
      out[j++] = '\\';
      out[j++] = 'r';
      break;
    case '\t':
      out[j++] = '\\';
      out[j++] = 't';
      break;
    default:
      if (ch < 0x20) {
        out[j++] = '\\';
        out[j++] = 'u';
        out[j++] = '0';
        out[j++] = '0';
        out[j++] = hex[ch >> 4];
        out[j++] = hex[ch & 0x0f];
      } else {
        out[j++] = (char)ch;
      }
      break;
    }
  }
  out[j] = '\0';
  return out;
}

static char *make_request(const char *prompt) {
  static const char prefix[] =
    "{\"model\":\"qwen3-0.6b\",\"messages\":[{\"role\":\"user\","
    "\"content\":\"";
  static const char suffix[] =
    "\"}],\"stream\":true,\"max_tokens\":32,\"temperature\":0}";
  char *escaped;
  char *request;
  size_t request_len;

  escaped = json_escape(prompt);
  if (escaped == NULL)
    return NULL;

  request_len = strlen(prefix) + strlen(escaped) + strlen(suffix) + 1;
  request = (char *)malloc(request_len);
  if (request == NULL) {
    free(escaped);
    return NULL;
  }

  strcpy(request, prefix);
  strcat(request, escaped);
  strcat(request, suffix);
  free(escaped);
  return request;
}

int main(int argc, char **argv) {
  const char *dll_path;
  const char *model_dir;
  const char *prompt;
  char *request_json;
  quick_ai_init_options opts;
  quick_ai_usage usage;
  quick_ai_ctx *ctx;
  quick_ai_status status;
  quick_ai_ctx_create_fn p_ctx_create;
  quick_ai_ctx_free_fn p_ctx_free;
  quick_ai_generate_fn p_generate;
  quick_ai_last_error_fn p_last_error;
  quick_ai_version_fn p_version;
#ifdef _WIN32
  HMODULE lib;
#else
  void *lib;
#endif

  if (argc < 3) {
    fprintf(stderr, "usage: %s <quick_ai.dll> <model_dir> [prompt]\n", argv[0]);
    return EXIT_FAILURE;
  }

  dll_path = argv[1];
  model_dir = argv[2];
  prompt = argc >= 4 ? argv[3] : "Write one short sentence about quick.ai.";

#ifdef _WIN32
  lib = LoadLibraryA(dll_path);
  if (lib == NULL) {
    fprintf(stderr, "failed to load %s (GetLastError=%lu)\n", dll_path,
            (unsigned long)GetLastError());
    return EXIT_FAILURE;
  }
#define LOAD_SYMBOL(name) (name##_fn)(void *) GetProcAddress(lib, #name)
#else
  lib = dlopen(dll_path, RTLD_NOW | RTLD_LOCAL);
  if (lib == NULL) {
    fprintf(stderr, "failed to load %s: %s\n", dll_path, dlerror());
    return EXIT_FAILURE;
  }
#define LOAD_SYMBOL(name) (name##_fn) dlsym(lib, #name)
#endif

  p_ctx_create = LOAD_SYMBOL(quick_ai_ctx_create);
  p_ctx_free = LOAD_SYMBOL(quick_ai_ctx_free);
  p_generate = LOAD_SYMBOL(quick_ai_generate);
  p_last_error = LOAD_SYMBOL(quick_ai_last_error);
  p_version = LOAD_SYMBOL(quick_ai_version);
  if (p_ctx_create == NULL || p_ctx_free == NULL || p_generate == NULL ||
      p_last_error == NULL || p_version == NULL) {
    fprintf(stderr, "required C ABI symbol is missing\n");
    return EXIT_FAILURE;
  }

  request_json = make_request(prompt);
  if (request_json == NULL) {
    fprintf(stderr, "failed to allocate request JSON\n");
    return EXIT_FAILURE;
  }

  opts.num_threads = 4;
  opts.enable_thinking = 0;
  ctx = NULL;
  status = p_ctx_create(model_dir, &opts, &ctx);
  if (status != 0) {
    fprintf(stderr, "quick_ai_ctx_create failed: %d: %s\n", status,
            p_last_error(NULL));
    free(request_json);
    return EXIT_FAILURE;
  }

  printf("%s\n", p_version());
  status = p_generate(ctx, request_json, print_token, NULL, &usage);
  printf("\n");
  if (status != 0) {
    fprintf(stderr, "quick_ai_generate failed: %d: %s\n", status,
            p_last_error(ctx));
    p_ctx_free(ctx);
    free(request_json);
    return EXIT_FAILURE;
  }

  printf("\nusage: prompt=%llu completion=%llu\n", usage.prompt_tokens,
         usage.completion_tokens);
  p_ctx_free(ctx);
  free(request_json);

#ifdef _WIN32
  FreeLibrary(lib);
#else
  dlclose(lib);
#endif
  return EXIT_SUCCESS;
}
