// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

// cutlass::bfloat16_t
// cutlass::half_t
#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
      using elem_type = cutlass::bfloat16_t;     \
      return __VA_ARGS__();                  \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                     \
    constexpr static int kHeadDim = 128;     \
    return __VA_ARGS__();                   \
  }()

#define MASK_SWITCH(MASK, ...)   \
  [&] {                                    \
    if (MASK == 0) {                       \
      constexpr static int mask = 0;       \
      return __VA_ARGS__();                \
    } else if (MASK == 4) {                \
      constexpr static int mask = 4;       \
      return __VA_ARGS__();                \
    }                                      \
  }()

#define WIDTH_SWITCH(WIDTH, ...)   \
  [&] {                                    \
    if (WIDTH == 64) {                     \
      constexpr static int width = 64;     \
      return __VA_ARGS__();                \
    }                                      \
  }()


// #define WIDTH_SWITCH(WIDTH, ...)   \
//   [&] {                                    \
//     if (WIDTH == 32) {                     \
//       constexpr static int width = 32;     \
//       return __VA_ARGS__();                \
//     } else if (WIDTH == 64) {              \
//       constexpr static int width = 64;     \
//       return __VA_ARGS__();                \
//     }                                      \
//   }()
