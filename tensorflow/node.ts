import * as tf from '@tensorflow/tfjs-node'
import { readFile } from 'fs/promises'
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
   * - "file://path/to/model.json"
   * - "http://localhost:8100/saved_models/yolo11n-pose_web_model"
   * - "https://domain.net/saved_models/yolo11n-pose_web_model/model.json"
   * */
  modelPath: string,
) {
  if (!modelPath.endsWith('/model.json')) {
    modelPath = modelPath + '/model.json'
  }
  if (!modelPath.includes('://')) {
    modelPath = 'file://' + modelPath
  }
  let model = tf.loadGraphModel(modelPath)

  let metadataPath = modelPath.replace(/model\.json$/, 'metadata.yaml')
  let metadata = loadText(metadataPath).then(text => parseMetadataYaml(text))

  return combineModelAndMetadata(await model, await metadata)
}

async function loadText(path: string): Promise<string> {
  if (path.startsWith('file://')) {
    return await readFile(path.slice('file://'.length), 'utf-8')
  }
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return await loadTextFromUrl(path)
  }
  throw new Error(`Unsupported path: "${path}"`)
}

export type ImageInput =
  | {
      /** path to image file */
      file: string
    }
  | {
      /**
       * input shape: [height, width, channels] or [batch, height, width, channels]
       *
       * the pixel values should be in the range of [0, 255]
       */
      tensor: tf.Tensor
    }
