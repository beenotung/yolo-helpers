import type { InferenceModel, Tensor } from '@tensorflow/tfjs'

export function getModelInputShape(model: InferenceModel) {
  if (model.inputs.length !== 1) {
    throw new Error(
      `model should have 1 input, but having ${model.inputs.length} inputs`,
    )
  }
  let shape = model.inputs[0].shape
  if (!shape) {
    throw new Error(`model input shape is not defined`)
  }
  let height = shape[1]
  let width = shape[2]
  return { height, width }
}

export function getImageSize(input: Tensor) {
  // [batch, height, width, channels]
  if (input.rank === 4) {
    let height = input.shape[1]
    let width = input.shape[2]
    return { height, width }
  }

  // [height, width, channels]
  if (input.rank === 3) {
    let height = input.shape[0]
    let width = input.shape[1]
    return { height, width }
  }

  throw new Error(`input rank should be 3 or 4, but got ${input.rank}`)
}

/** normalize color and resize/expand into shape: [batch, height, width, channels] */
export function preprocessInput(
  /**
   * input shape: [height, width, channels] or [batch, height, width, channels]
   *
   * the pixel values should be in the range of [0, 255]
   */
  input: Tensor,
  input_shape: { height: number; width: number },
) {
  // expand batch dimension if input is 2D
  if (input.rank === 3) {
    input = input.expandDims()
  }

  // resize input to input_shape if necessary
  let input_height = input.shape[1]
  let input_width = input.shape[2]
  if (
    input_width !== input_shape.width ||
    input_height !== input_shape.height
  ) {
    input = input.resizeBilinear([input_shape.height, input_shape.width])
  }

  // normalize to 0..1
  input = input.div(255.0)

  return input
}

export type ModelMetadata = {
  names?: string[]
}
/**
 *
 * example:
 * ```
 * version: 8.3.83
 * task: segment
 * batch: 1
 * imgsz:
 * - 640
 * - 640
 * names:
 *   0: person
 *   1: bicycle
 *   2: car
 * args:
 *   batch: 1
 *   half: false
 *   int8: false
 *   nms: false
 * ```
 *
 */
export function parseMetadataYaml(text: string): ModelMetadata {
  let lines = text
    .split('\n')
    .map(line => line.replaceAll('\r', ''))
    .filter(line => !line.trim().startsWith('#'))
  let start = lines.indexOf('names:')
  if (start == -1) {
    return {}
  }
  start++
  let end = lines.findIndex(
    (line, index) => index >= start && !line.startsWith(' '),
  )
  if (end == -1) {
    end = lines.length
  }
  let names: string[] = []
  lines.slice(start, end).forEach(line => {
    line = line.split('#')[0]
    // e.g. "9: traffic light"
    let index = line.indexOf(':')
    if (index == -1) {
      throw new Error(
        `failed to parse class name, line: ${JSON.stringify(line)}`,
      )
    }
    let cls_index = +line.slice(0, index).trim()
    let cls_name = line.slice(index + 1).trim()
    if (!Number.isInteger(cls_index)) {
      throw new Error(
        `failed to parse class index, line: ${JSON.stringify(line)}`,
      )
    }
    names[cls_index] = cls_name
  })
  return {
    names,
  }
}

export function combineModelAndMetadata<T extends InferenceModel>(
  model: T,
  metadata: ModelMetadata,
) {
  return Object.assign(model, {
    class_names: metadata.names,
  })
}

export async function loadTextFromUrl(url: string): Promise<string> {
  let res = await fetch(url)
  if (!res.ok) {
    throw new Error(res.statusText || `status: ${res.status}`)
  }
  let text = await res.text()
  return text
}

export function calculateNumOfOutputBoxes(width: number, height: number) {
  if (width % 32 !== 0) {
    width += 32 - (width % 32)
    console.warn('width is not multiplier of 32, auto adjusted to:', width)
  }
  if (height % 32 !== 0) {
    height += 32 - (height % 32)
    console.warn('height is not multiplier of 32, auto adjusted to:', height)
  }
  let num_boxes =
    (width / 8) * (height / 8) +
    (width / 16) * (height / 16) +
    (width / 32) * (height / 32)
  return num_boxes
}
