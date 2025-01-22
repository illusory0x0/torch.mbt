// =====================================
// WARNING: very unstable API, for internal use only
// =====================================

#ifndef moonbit_h_INCLUDED
#define moonbit_h_INCLUDED

#ifdef MOONBIT_NATIVE_NO_SYS_HEADER
#include "moonbit-fundamental.h"
#else
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#endif

enum moonbit_block_kind {
  // 0 => regular block
  moonbit_BLOCK_KIND_REGULAR = 0,
  // 1 => array of pointers
  moonbit_BLOCK_KIND_REF_ARRAY = 1,
  // 2 => array of immediate value/string/bytes
  moonbit_BLOCK_KIND_VAL_ARRAY = 2
  // 3 => reserved
};

struct moonbit_array_header {
  enum moonbit_block_kind kind : 2;
  // The size of array element is [2^object_size_shift] bytes
  unsigned int object_size_shift : 2;
  // The number of elements
  unsigned int len : 28;
};

struct moonbit_object {
  int32_t rc;
  union {
    struct {
      enum moonbit_block_kind kind : 2;
      /* The 22 length bits are separated into two 11 bit parts:
 
         - [ptr_field_offset] is the offset of the first pointer field,
           counted by the number of 32-bit words.
           The object header itself is also included.

         - [n_ptr_fields] is the number of pointer fields

         We rearrange the layout of all pointer fields
         so that all pointer fields are placed at the end.
         This make it easy for the runtime to enumerate all pointer fields in an object,
         without any static type information.

         The total length of the object can be reconstructed via:

           ptr_field_offset * 4 + n_ptr_fields * sizeof(void*)
      */
      unsigned int ptr_field_offset : 11;
      unsigned int n_ptr_fields : 11;
      // For blocks, we steal 8 bits from the length to represent enum tag
      unsigned int tag : 8;
    } block;
    /* For array/string/bytes, there is no need to store tag & bitmap,
       so we may use all 30 remaining bits to store length */
    struct moonbit_array_header arr;
  };
};

struct moonbit_object *moonbit_malloc(size_t size);
void moonbit_incref(void *obj);
extern "C" void moonbit_decref(void *obj);

struct moonbit_string {
  struct moonbit_object header;
  uint16_t data[0];
};

struct moonbit_bytes {
  struct moonbit_object header;
  uint8_t data[0];
};

struct moonbit_int32_array {
  struct moonbit_object header;
  int32_t data[0];
};

struct moonbit_ref_array {
  struct moonbit_object header;
  void *data[0];
};

struct moonbit_int64_array {
  struct moonbit_object header;
  int64_t data[0];
};

struct moonbit_double_array {
  struct moonbit_object header;
  double data[0];
};

struct moonbit_float_array {
  struct moonbit_object header;
  float data[0];
};


struct moonbit_string *moonbit_make_string(int size, int value);
extern "C" struct moonbit_bytes *moonbit_make_bytes(int size, int value);
struct moonbit_int32_array *moonbit_make_int32_array(int len, int32_t value);
struct moonbit_ref_array *moonbit_make_ref_array(int len, void *value);
struct moonbit_int64_array *moonbit_make_int64_array(int len, int64_t value);
struct moonbit_double_array *moonbit_make_double_array(int len, double value);
struct moonbit_float_array *moonbit_make_float_array(int len, float value);

#endif // moonbit_h_INCLUDED
