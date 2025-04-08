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
  task?: 'detect' | 'pose' | 'segment' | string
  class_names?: string[]
  keypoints?: number
  visibility?: boolean
}
/**
 * example of segmentation model:
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
 * example of pose model:
 * ```
 * task: pose
 * kpt_shape:
 * - 17
 * - 3
 * ```
 */
export function parseMetadataYaml(text: string): ModelMetadata {
  let lines = parseLines(text)

  // e.g. "task: pose" -> "pose"
  let task = lines
    .find(line => line.startsWith('task:'))
    ?.split(':')[1]
    .trim()

  /**
   * e.g.
   * ```
   * kpt_shape:
   * - 17 # number of keypoints
   * - 3  # number of dimensions per keypoint, 2 for {x,y}, 3 for {x,y,visibility}
   * ```
   */
  let index = lines.indexOf('kpt_shape:')
  let keypoints = index == -1 ? undefined : parseIntFromLine(lines[index + 1])
  let visibility =
    index == -1 ? undefined : parseIntFromLine(lines[index + 2]) == 3

  // e.g. "names:"
  index = lines.indexOf('names:')
  let class_names: string[] = []
  for (let i = index + 1; i < lines.length; i++) {
    // e.g. "  0: person" -> "person"
    let line = lines[i].trim()
    let idx = parseInt(line)
    let name = line.replace(String(idx), '').replace(':', '').trim()
    class_names[idx] = name
  }

  return { task, class_names, keypoints, visibility }
}

// e.g. "  - 17" -> 17
function parseIntFromLine(line: string) {
  return parseInt(line.replace('-', '').trim())
}

// skip empty line and comment lines starting with '#'
function parseLines(text: string) {
  return text
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0 && !line.startsWith('#'))
}

export type ModelWithMetadata<T extends InferenceModel> = T & ModelMetadata

export function combineModelAndMetadata<T extends InferenceModel>(
  model: T,
  metadata: ModelMetadata,
): ModelWithMetadata<T> {
  return Object.assign(model, metadata)
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
