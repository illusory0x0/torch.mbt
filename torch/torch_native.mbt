///|
fn tch_for_poc() -> Bytes {
  return for_poc_ffi()
}

///|
extern "C" fn for_poc_ffi() -> Bytes = "for_poc"
