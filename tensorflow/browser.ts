import * as tf from '@tensorflow/tfjs'
import {
  combineModelAndMetadata,
  loadTextFromUrl,
  parseMetadataYaml,
} from './common'

export async function loadYoloModel(
  /**
   * Can be with or without `/model.json`.
   *
   * Examples:
   * - "./saved_model/yolo11n-pose_web_model/model.json"
   * - "./saved_model/yolo11n-pose_web_model"
   * - "http://localhost:8100/saved_models/yolo11n-pose_web_model/model.json"
   * - "https://domain.net/saved_models/yolo11n-pose_web_model"
   * - "indexeddb://yolo11n-pose_web_model"
   * */
  modelUrl: string,
) {
  if (!modelUrl.endsWith('/model.json')) {
    modelUrl = modelUrl + '/model.json'
  }
  if (!modelUrl.includes('://')) {
    let parts = location.pathname.split('/')
    if (parts.length > 1) {
      parts.splice(parts.length - 1, 1)
    }
    let prefix = location.origin + parts.join('/')
    if (!modelUrl.startsWith('/')) {
      modelUrl = '/' + modelUrl
    }
    modelUrl = prefix + modelUrl
  }
  let model = tf.loadGraphModel(modelUrl)

  let metadataUrl = modelUrl.replace(/model\.json$/, 'metadata.yaml')
  let metadata = loadTextFromUrl(metadataUrl).then(text =>
    parseMetadataYaml(text),
  )

  return combineModelAndMetadata(await model, await metadata)
}

export type ImageInput =
  | { pixels: Parameters<typeof tf.browser.fromPixels>[0] }
  | {
      /**
       * input shape: [height, width, channels] or [batch, height, width, channels]
       *
       * the pixel values should be in the range of [0, 255]
       */
      tensor: tf.Tensor
    }
