import type * as tf_type from '@tensorflow/tfjs'
import { BoundingBox } from '../yolo-box/common'

/** [height, width, num_channels] -> 0 for background, 1 for object */
export type Mask = number[][]
export type BoundingBoxWithMaskCoefficients = BoundingBox & {
  /** 32 coefficients of mask */
  mask_coefficients: number[]
}

/**
 * output shape: [batch, box]
 *
 * Array of batches, each containing array of detected bounding boxes with masks coefficients and masks
 * */
export type SegmentResult = {
  bounding_boxes: BoundingBoxWithMaskCoefficients[]
  /** e.g. [mask_height, mask_width, 32] for 32 channels of masks */
  masks: Mask[]
}[]

export type ImageSize = { width: number; height: number }

export type DecodeSegmentArgs = {
  /**
   * tensorflow runtime:
   * - browser: `import * as tf from '@tensorflow/tfjs'`
   * - nodejs: `import * as tf from '@tensorflow/tfjs-node'`
   */
  tf: typeof tf_type
  /** e.g. `1` for single class */
  num_classes: number
  /**
   * Number of channels in segmentation mask
   * default: `32`
   */
  num_channels?: number

  /** batched predict result, e.g. 1x116x8400 */
  output_boxes: number[][][]
  /** batched predict result, e.g. 1x160x160x32 */
  output_masks: number[][][][]
  /**
   * Number of boxes to return using non-max suppression.
   * If not provided, all boxes will be returned
   *
   * e.g. `1` for only selecting the bounding box with highest confidence.
   */
  maxOutputSize?: number
  /**
   * the threshold for deciding whether boxes overlap too much with respect to IOU.
   *
   * default: `0.5`
   */
  iouThreshold?: number
  /**
   * the threshold for deciding whether a box is a valid detection.
   *
   * default: `-Infinity`
   */
  scoreThreshold?: number
} & (
  | {
      /** default: `{ width: 640, height: 640 }` */
      input_shape: ImageSize
    }
  | {
      /** default: input_shape / 4 */
      mask_shape: ImageSize
    }
)

function getMaskShape(args: DecodeSegmentArgs): ImageSize {
  if ('mask_shape' in args) return args.mask_shape
  if ('input_shape' in args) {
    return {
      width: args.input_shape.width / 4,
      height: args.input_shape.height / 4,
    }
  }
  throw new Error('missing mask_shape or input_shape')
}

/**
 * tensorflow output: boxes [batch, features, channel] and masks [batch, height, width, channel]
 *
 * box features:
 * - 4: x, y, width, height
 * - num_classes: class confidence
 * - 32: channel coefficients
 *
 * segmentation mask:
 * - 0 for background, 1 for object
 * - 32 channels, correspond to the 32 channel coefficients in the bounding box
 *
 * The x, y, width, height are in pixel unit, NOT normalized in the range of [0, 1].
 * The the pixel units are scaled to the input_shape.
 *
 * The confidence are already normalized between 0 to 1.
 */
export async function decodeSegment(
  args: DecodeSegmentArgs,
): Promise<SegmentResult> {
  let { tf, num_classes, maxOutputSize, iouThreshold, scoreThreshold } = args
  let num_channels = args.num_channels ?? 32

  let { width: mask_width, height: mask_height } = getMaskShape(args)

  let boxes_length = 4 + num_classes + num_channels

  // e.g. 1x116x8400
  let batches_boxes = args.output_boxes

  if (batches_boxes[0].length === 0) {
    // no a single batch
    return []
  }
  if (batches_boxes[0].length !== boxes_length) {
    throw new Error(`boxes_data[batch].length must be ${boxes_length}`)
  }
  let num_boxes = batches_boxes[0][0].length

  // e.g. 1x160x160x32
  let batches_masks = args.output_masks
  if (batches_masks[0].length !== mask_height) {
    throw new Error(`masks_data[batch].length must be ${mask_height}`)
  }
  if (batches_masks[0][0].length !== mask_width) {
    throw new Error(`masks_data[batch][y].length must be ${mask_width}`)
  }
  if (batches_masks[0][0][0].length !== num_channels) {
    throw new Error(`masks_data[batch][y][x].length must be ${num_channels}`)
  }

  if (batches_boxes.length !== batches_masks.length) {
    throw new Error('boxes_data and masks_data must have the same length')
  }

  let result: SegmentResult = []
  let batch_size = batches_boxes.length
  for (let batch = 0; batch < batch_size; batch++) {
    // 116x8400
    let batch_boxes = batches_boxes[batch]

    // 160x160x32
    let batch_masks = batches_masks[batch]

    let boxes: [x1: number, y1: number, x2: number, y2: number][] = []

    let cls_scores: number[] = []
    let cls_indices: number[] = []

    for (let box_index = 0; box_index < num_boxes; box_index++) {
      let x = batch_boxes[0][box_index]
      let y = batch_boxes[1][box_index]
      let width = batch_boxes[2][box_index]
      let height = batch_boxes[3][box_index]

      let x1 = x - width / 2
      let y1 = y - height / 2
      let x2 = x + width / 2
      let y2 = y + height / 2

      let cls_score = batch_boxes[4][box_index]
      let cls_index = 0
      for (let i = 1; i < num_classes; i++) {
        let score = batch_boxes[4 + i][box_index]
        if (score > cls_score) {
          cls_score = score
          cls_index = i
        }
      }

      boxes.push([x1, y1, x2, y2])

      cls_scores.push(cls_score)
      cls_indices.push(cls_index)
    }

    let box_indices: number[]
    if (maxOutputSize) {
      let box_indices_tensor = await tf.image.nonMaxSuppressionAsync(
        boxes,
        cls_scores,
        maxOutputSize,
        iouThreshold,
        scoreThreshold,
      )
      box_indices = await box_indices_tensor.array()
      box_indices_tensor.dispose()
    } else {
      box_indices = Array.from({ length: num_boxes }, (_, i) => i)
    }

    let bounding_boxes: BoundingBoxWithMaskCoefficients[] = []
    for (let box_index of box_indices) {
      let x = batch_boxes[0][box_index]
      let y = batch_boxes[1][box_index]
      let width = batch_boxes[2][box_index]
      let height = batch_boxes[3][box_index]
      let class_index = cls_indices[box_index]
      let confidence = batch_boxes[4 + class_index][box_index]
      let all_confidences: number[] = new Array(num_classes)
      for (let i = 0; i < num_classes; i++) {
        all_confidences[i] = batch_boxes[4 + i][box_index]
      }
      let mask_coefficients: number[] = new Array(num_channels)
      for (let i = 0; i < num_channels; i++) {
        mask_coefficients[i] = batch_boxes[4 + num_classes + i][box_index]
      }
      bounding_boxes.push({
        x,
        y,
        width,
        height,
        class_index,
        confidence,
        all_confidences,
        mask_coefficients,
      })
    }
    result.push({
      bounding_boxes,
      masks: batch_masks,
    })
  }

  return result
}

/**
 * Sync version of `decodeSegment`.
 */
export function decodeSegmentSync(args: DecodeSegmentArgs): SegmentResult {
  let { tf, num_classes, maxOutputSize, iouThreshold, scoreThreshold } = args
  let num_channels = args.num_channels ?? 32

  let { width: mask_width, height: mask_height } = getMaskShape(args)

  let boxes_length = 4 + num_classes + num_channels

  // e.g. 1x116x8400
  let batches_boxes = args.output_boxes

  if (batches_boxes[0].length === 0) {
    // no a single batch
    return []
  }
  if (batches_boxes[0].length !== boxes_length) {
    throw new Error(`boxes_data[batch].length must be ${boxes_length}`)
  }
  let num_boxes = batches_boxes[0][0].length

  // e.g. 1x160x160x32
  let batches_masks = args.output_masks
  if (batches_masks[0].length !== mask_height) {
    throw new Error(`masks_data[batch].length must be ${mask_height}`)
  }
  if (batches_masks[0][0].length !== mask_width) {
    throw new Error(`masks_data[batch][y].length must be ${mask_width}`)
  }
  if (batches_masks[0][0][0].length !== num_channels) {
    throw new Error(`masks_data[batch][y][x].length must be ${num_channels}`)
  }

  if (batches_boxes.length !== batches_masks.length) {
    throw new Error('boxes_data and masks_data must have the same length')
  }

  let result: SegmentResult = []
  let batch_size = batches_boxes.length
  for (let batch = 0; batch < batch_size; batch++) {
    // 116x8400
    let batch_boxes = batches_boxes[batch]

    // 160x160x32
    let batch_masks = batches_masks[batch]

    let boxes: [x1: number, y1: number, x2: number, y2: number][] = []

    let cls_scores: number[] = []
    let cls_indices: number[] = []

    for (let box_index = 0; box_index < num_boxes; box_index++) {
      let x = batch_boxes[0][box_index]
      let y = batch_boxes[1][box_index]
      let width = batch_boxes[2][box_index]
      let height = batch_boxes[3][box_index]

      let x1 = x - width / 2
      let y1 = y - height / 2
      let x2 = x + width / 2
      let y2 = y + height / 2

      let cls_score = batch_boxes[4][box_index]
      let cls_index = 0
      for (let i = 1; i < num_classes; i++) {
        let score = batch_boxes[4 + i][box_index]
        if (score > cls_score) {
          cls_score = score
          cls_index = i
        }
      }

      boxes.push([x1, y1, x2, y2])

      cls_scores.push(cls_score)
      cls_indices.push(cls_index)
    }

    let box_indices: number[]
    if (maxOutputSize) {
      box_indices = tf.tidy(() => {
        let box_indices_tensor = tf.image.nonMaxSuppression(
          boxes,
          cls_scores,
          maxOutputSize,
          iouThreshold,
          scoreThreshold,
        )
        return box_indices_tensor.arraySync()
      })
    } else {
      box_indices = Array.from({ length: num_boxes }, (_, i) => i)
    }

    let bounding_boxes: BoundingBoxWithMaskCoefficients[] = []
    for (let box_index of box_indices) {
      let x = batch_boxes[0][box_index]
      let y = batch_boxes[1][box_index]
      let width = batch_boxes[2][box_index]
      let height = batch_boxes[3][box_index]
      let class_index = cls_indices[box_index]
      let confidence = batch_boxes[4 + class_index][box_index]
      let all_confidences: number[] = new Array(num_classes)
      for (let i = 0; i < num_classes; i++) {
        all_confidences[i] = batch_boxes[4 + i][box_index]
      }
      let mask_coefficients: number[] = new Array(num_channels)
      for (let i = 0; i < num_channels; i++) {
        mask_coefficients[i] = batch_boxes[4 + num_classes + i][box_index]
      }
      bounding_boxes.push({
        x,
        y,
        width,
        height,
        class_index,
        confidence,
        all_confidences,
        mask_coefficients,
      })
    }
    result.push({
      bounding_boxes,
      masks: batch_masks,
    })
  }

  return result
}

/**
 * @description final mask = mask coefficients * mask channels
 */
export function combineMask(
  bounding_box: BoundingBoxWithMaskCoefficients,
  /** e.g. [mask_height, mask_width, 32] for 32 channels of masks */
  masks: Mask[],
) {
  let mask_coefficients = bounding_box.mask_coefficients

  let mask_height = masks.length
  let mask_width = masks[0].length
  let num_channels = masks[0][0].length

  if (num_channels != mask_coefficients.length) {
    throw new Error(
      `expect ${num_channels} mask coefficients, but got ${mask_coefficients.length}`,
    )
  }

  let final_mask: number[][] = new Array(mask_height)
  for (let h = 0; h < mask_height; h++) {
    final_mask[h] = new Array(mask_width)
    for (let w = 0; w < mask_width; w++) {
      let acc = 0
      for (let i = 0; i < num_channels; i++) {
        acc += mask_coefficients[i] * masks[h][w][i]
      }
      acc = sigmoid(acc)
      final_mask[h][w] = acc
    }
  }
  return final_mask
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

export type Rect = { left: number; top: number; right: number; bottom: number }

export function hasOverlap(a: Rect, b: Rect): boolean {
  return !(
    a.right < b.left ||
    a.left > b.right ||
    a.bottom < b.top ||
    a.top > b.bottom
  )
}
