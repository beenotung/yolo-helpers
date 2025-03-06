export type ImageResult = {
  /** class index with highest confidence */
  class_index: number
  /** confidence of the class with highest confidence */
  confidence: number
  /** confidence of all classes */
  all_confidences: number[]
}

/**
 * output shape: [batch]
 *
 * Array of batches, each containing array of confidence for each classes
 * */
export type ClassifyResult = ImageResult[]

export type DecodeClassifyArgs = {
  /** e.g. `1` for single class */
  num_classes: number
  /** batched predict result, e.g. 1x80 */
  output: number[][]
}

export function decodeClassify(args: DecodeClassifyArgs): ClassifyResult {
  let { num_classes } = args

  // e.g. 1x80
  let batches = args.output

  if (batches[0].length === 0) {
    // no a single batch
    return []
  }
  if (batches[0].length !== num_classes) {
    throw new Error(`data[batch].length must be ${num_classes}`)
  }

  let result: ClassifyResult = []
  for (let all_confidences of batches) {
    // e.g. 80 numbers

    let class_index = 0
    let class_confidence = all_confidences[0]
    for (let i = 1; i < num_classes; i++) {
      let confidence = all_confidences[i]
      if (confidence > class_confidence) {
        class_confidence = confidence
        class_index = i
      }
    }
    result.push({
      all_confidences,
      class_index,
      confidence: class_confidence,
    })
  }
  return result
}
