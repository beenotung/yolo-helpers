import * as tf from '@tensorflow/tfjs'
import { readFileSync } from 'fs'
import { join } from 'path'
import { parseMetadataYaml } from '../tensorflow/common'

type WeightManifestGroup = {
  paths: string[]
  weights: tf.io.WeightsManifestEntry[]
}

function getModelDir(input: string) {
  return input.endsWith('model.json') ? input.replace(/\/?model\.json$/, '') : input
}

function readTextIfExists(file: string) {
  try {
    return readFileSync(file, 'utf8')
  } catch {
    return undefined
  }
}

function toOutputArray(output: tf.Tensor | tf.Tensor[]) {
  return Array.isArray(output) ? output : [output]
}

function shapeOf(tensor: tf.Tensor) {
  return `[${tensor.shape.join(',')}] ${tensor.dtype}`
}

async function loadLocalGraphModel(modelDir: string) {
  let modelJson = JSON.parse(readFileSync(join(modelDir, 'model.json'), 'utf8'))
  let manifest = modelJson.weightsManifest as WeightManifestGroup[]
  let weightSpecs = manifest.flatMap(group => group.weights)
  let buffers: Buffer[] = []
  for (let group of manifest) {
    for (let path of group.paths) {
      buffers.push(readFileSync(join(modelDir, path)))
    }
  }
  let data = Buffer.concat(buffers)
  let weightData = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
  let handler = tf.io.fromMemory({
    modelTopology: modelJson.modelTopology,
    weightSpecs,
    weightData,
    format: modelJson.format,
    generatedBy: modelJson.generatedBy,
    convertedBy: modelJson.convertedBy,
    signature: modelJson.signature,
  })
  return tf.loadGraphModel(handler)
}

async function main() {
  let modelArg = process.argv[2] ?? 'yolo26/models/yolo26n_web_model'
  let modelDir = getModelDir(modelArg)
  let metadataText = readTextIfExists(join(modelDir, 'metadata.yaml'))
  let metadata = metadataText ? parseMetadataYaml(metadataText) : {}

  console.log('model:', modelDir)
  console.log('metadata:', {
    task: metadata.task,
    end2end: metadata.end2end,
    nms: metadata.nms,
    box_output_format: metadata.box_output_format,
    classify_output_format: metadata.classify_output_format,
    pose_output_format: metadata.pose_output_format,
    segment_output_format: metadata.segment_output_format,
    class_names: metadata.class_names?.length,
    keypoints: metadata.keypoints,
    visibility: metadata.visibility,
  })

  let model = await loadLocalGraphModel(modelDir)
  console.log(
    'inputs:',
    model.inputs.map(input => ({
      name: input.name,
      shape: input.shape,
      dtype: input.dtype,
    })),
  )
  console.log(
    'declared outputs:',
    model.outputs.map(output => ({
      name: output.name,
      shape: output.shape,
      dtype: output.dtype,
    })),
  )

  if (model.inputs.length !== 1 || !model.inputs[0].shape) {
    console.log('execute outputs: skipped because model input shape is not supported')
    model.dispose()
    return
  }

  let inputShape = model.inputs[0].shape.map((size, index) =>
    size == null || size < 1 ? (index === 0 ? 1 : 640) : size,
  )
  let input = tf.zeros(inputShape)
  let output = model.predict(input) as tf.Tensor | tf.Tensor[]
  let outputs = toOutputArray(output)
  console.log('execute outputs:', outputs.map(shapeOf))
  outputs.forEach(tensor => tensor.dispose())
  input.dispose()
  model.dispose()
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})
