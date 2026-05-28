use burn::{
    module::{Module, Quantizer},
    prelude::*,
    tensor::{
        ops::QuantizedTensor,
        quantization::{Calibration, QTensorPrimitive, QuantLevel, QuantParam, QuantValue},
    },
};

use crate::model::TinyLm;

pub fn quantize_int8_weights<B: Backend>(model: TinyLm<B>) -> TinyLm<B> {
    let scheme = <QuantizedTensor<B> as QTensorPrimitive>::default_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::Tensor)
        .with_param(QuantParam::F32);

    let mut quantizer = Quantizer {
        calibration: Calibration::MinMax,
        scheme,
    };

    model.quantize_weights(&mut quantizer)
}
