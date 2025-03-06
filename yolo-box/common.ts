import type * as tf_type from '@tensorflow/tfjs'

export type BoundingBox = {
  /** center x of bounding box in px */
  x: number
  /** center y of bounding box in px */
  y: number
  /** width of bounding box in px */
  width: number
  /** height of bounding box in px */
  height: number
  /** class index with highest confidence */
  class_index: number
  /** confidence of the class with highest confidence */
  confidence: number
  /** confidence of all classes */
  all_confidences: number[]
}

/**
 * output shape: [batch, box]
 *
 * Array of batches, each containing array of detected bounding boxes
 * */
export type BoxResult = BoundingBox[][]

export type DecodeBoxArgs = {
  /**
   * tensorflow runtime:
   * - browser: `import * as tf from '@tensorflow/tfjs'`
   * - nodejs: `import * as tf from '@tensorflow/tfjs-node'`
   */
  tf: typeof tf_type
  /** e.g. `1` for single class */
  num_classes: number
  /** batched predict result, e.g. 1x84x8400 */
  output: number[][][]
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
}

/**
 * tensorflow output: [batch, features, instances]
 * features:
 * - 4: x, y, width, height
 * - num_classes: class confidence
 *
 * e.g. 1x84x8400 for 1 batch of 8400 instances with 80 classes
 *
 * The confidence are already normalized between 0 to 1.
 */
export async function decodeBox(args: DecodeBoxArgs): Promise<BoxResult> {
  let { tf, num_classes, maxOutputSize, iouThreshold, scoreThreshold } = args
  let length = 4 + num_classes

  // e.g. 1x84x8400
  let batches = args.output

  if (batches[0].length === 0) {
    // no a single batch
    return []
  }
  if (batches[0].length !== length) {
    throw new Error(`data[batch].length must be ${length}`)
  }

  let num_boxes = batches[0][0].length

  let result: BoxResult = []
  for (let batch of batches) {
    // e.g. 84x8400

    let boxes: [x1: number, y1: number, x2: number, y2: number][] = []
    let scores: number[] = []
    let cls_indices: number[] = []
    for (let box_index = 0; box_index < num_boxes; box_index++) {
      let x = batch[0][box_index]
      let y = batch[1][box_index]
      let width = batch[2][box_index]
      let height = batch[3][box_index]

      let x1 = x - width / 2
      let y1 = y - height / 2
      let x2 = x + width / 2
      let y2 = y + height / 2

      let box_score = batch[4][box_index]
      let cls_index = 0
      for (let i = 1; i < num_classes; i++) {
        let cls_score = batch[4 + i][box_index]
        if (cls_score > box_score) {
          box_score = cls_score
          cls_index = i
        }
      }

      boxes.push([x1, y1, x2, y2])
      scores.push(box_score)
      cls_indices.push(cls_index)
    }

    let box_indices: number[]
    if (maxOutputSize) {
      let box_indices_tensor = await tf.image.nonMaxSuppressionAsync(
        boxes,
        scores,
        maxOutputSize,
        iouThreshold,
        scoreThreshold,
      )
      box_indices = await box_indices_tensor.array()
      box_indices_tensor.dispose()
    } else {
      box_indices = Array.from({ length: num_boxes }, (_, i) => i)
    }

    let bounding_boxes: BoundingBox[] = []
    for (let box_index of box_indices) {
      let x = batch[0][box_index]
      let y = batch[1][box_index]
      let width = batch[2][box_index]
      let height = batch[3][box_index]
      let class_index = cls_indices[box_index]
      let confidence = batch[4 + class_index][box_index]
      let all_confidences: number[] = new Array(num_classes)
      for (let i = 0; i < num_classes; i++) {
        all_confidences[i] = batch[4 + i][box_index]
      }
      bounding_boxes.push({
        x,
        y,
        width,
        height,
        class_index,
        confidence,
        all_confidences,
      })
    }
    result.push(bounding_boxes)
  }
  return result
}

/**
 * Sync version of `decodeBox`.
 */
export function decodeBoxSync(args: DecodeBoxArgs): BoxResult {
  let { tf, num_classes, maxOutputSize, iouThreshold, scoreThreshold } = args
  let length = 4 + num_classes

  // e.g. 1x84x8400
  let batches = args.output

  if (batches[0].length === 0) {
    // no a single batch
    return []
  }
  if (batches[0].length !== length) {
    throw new Error(`data[batch].length must be ${length}`)
  }

  let num_boxes = batches[0][0].length

  let result: BoxResult = []
  for (let batch of batches) {
    // e.g. 84x8400

    let boxes: [x1: number, y1: number, x2: number, y2: number][] = []
    let scores: number[] = []
    let cls_indices: number[] = []
    for (let box_index = 0; box_index < num_boxes; box_index++) {
      let x = batch[0][box_index]
      let y = batch[1][box_index]
      let width = batch[2][box_index]
      let height = batch[3][box_index]

      let x1 = x - width / 2
      let y1 = y - height / 2
      let x2 = x + width / 2
      let y2 = y + height / 2

      let box_score = batch[4][box_index]
      let cls_index = 0
      for (let i = 1; i < num_classes; i++) {
        let cls_score = batch[4 + i][box_index]
        if (cls_score > box_score) {
          box_score = cls_score
          cls_index = i
        }
      }

      boxes.push([x1, y1, x2, y2])
      scores.push(box_score)
      cls_indices.push(cls_index)
    }

    let box_indices: number[]
    if (maxOutputSize) {
      box_indices = tf.tidy(() =>
        tf.image
          .nonMaxSuppression(
            boxes,
            scores,
            maxOutputSize,
            iouThreshold,
            scoreThreshold,
          )
          .arraySync(),
      )
    } else {
      box_indices = Array.from({ length: num_boxes }, (_, i) => i)
    }

    let bounding_boxes: BoundingBox[] = []
    for (let box_index of box_indices) {
      let x = batch[0][box_index]
      let y = batch[1][box_index]
      let width = batch[2][box_index]
      let height = batch[3][box_index]
      let class_index = cls_indices[box_index]
      let confidence = batch[4 + class_index][box_index]
      let all_confidences: number[] = new Array(num_classes)
      for (let i = 0; i < num_classes; i++) {
        all_confidences[i] = batch[4 + i][box_index]
      }
      bounding_boxes.push({
        x,
        y,
        width,
        height,
        class_index,
        confidence,
        all_confidences,
      })
    }
    result.push(bounding_boxes)
  }
  return result
}
