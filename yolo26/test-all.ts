import * as tf from '@tensorflow/tfjs'
import {
  classifyImage,
  combineMask,
  detectBox,
  detectPose,
  detectSegment,
  drawBox,
  hasOverlap,
  loadYoloModel,
} from '../browser'
import type { BoxOutputFormat } from '../yolo-box/common'
import type { PoseOutputFormat } from '../yolo-pose/common'
import type { SegmentOutputFormat } from '../yolo-segment/common'

type Task = 'classify' | 'detect' | 'pose' | 'segment'
type OutputFormat = 'auto' | 'yolo' | 'yolo26' | 'end2end'

const taskInput = document.querySelector<HTMLSelectElement>('#task')!
const modelUrlInput = document.querySelector<HTMLInputElement>('#modelUrl')!
const fileInput = document.querySelector<HTMLInputElement>('#file')!
const scoreThresholdInput =
  document.querySelector<HTMLInputElement>('#scoreThreshold')!
const iouThresholdInput =
  document.querySelector<HTMLInputElement>('#iouThreshold')!
const maxOutputSizeInput =
  document.querySelector<HTMLInputElement>('#maxOutputSize')!
const numClassesInput = document.querySelector<HTMLInputElement>('#numClasses')!
const numKeypointsInput =
  document.querySelector<HTMLInputElement>('#numKeypoints')!
const numChannelsInput =
  document.querySelector<HTMLInputElement>('#numChannels')!
const outputFormatInput =
  document.querySelector<HTMLSelectElement>('#outputFormat')!
const loadButton = document.querySelector<HTMLButtonElement>('#load')!
const runButton = document.querySelector<HTMLButtonElement>('#run')!
const status = document.querySelector<HTMLDivElement>('#status')!
const canvas = document.querySelector<HTMLCanvasElement>('#canvas')!
const context = canvas.getContext('2d')!
const metadata = document.querySelector<HTMLPreElement>('#metadata')!
const results = document.querySelector<HTMLOListElement>('#results')!

let model: Awaited<ReturnType<typeof loadYoloModel>> | undefined
let modelUrl = ''
let image = new Image()
let imageReady = false

function setStatus(text: string) {
  status.textContent = text
}

function setResults(items: string[]) {
  results.replaceChildren(
    ...items.map(text => {
      let item = document.createElement('li')
      item.textContent = text
      return item
    }),
  )
}

function parseNumber(input: HTMLInputElement, fallback: number) {
  let value = Number(input.value)
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function parseOptionalNumber(input: HTMLInputElement) {
  let value = Number(input.value)
  return Number.isFinite(value) && value > 0 ? value : undefined
}

function getOutputFormat(): OutputFormat | undefined {
  let value = outputFormatInput.value as OutputFormat
  return value === 'auto' ? undefined : value
}

function drawImageToCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height)
  if (imageReady) {
    context.drawImage(image, 0, 0, canvas.width, canvas.height)
  }
}

function showModelMetadata() {
  if (!model) {
    metadata.textContent = 'No model loaded.'
    return
  }
  metadata.textContent = JSON.stringify(
    {
      task: model.task,
      input: model.inputs[0]?.shape,
      outputs: model.outputs.map(output => ({
        name: output.name,
        shape: output.shape,
      })),
      class_names: model.class_names?.length,
      keypoints: model.keypoints,
      visibility: model.visibility,
      end2end: model.end2end,
      nms: model.nms,
      box_output_format: model.box_output_format,
      classify_output_format: model.classify_output_format,
      pose_output_format: model.pose_output_format,
      segment_output_format: model.segment_output_format,
    },
    null,
    2,
  )
}

function getInputShape() {
  if (!model?.inputs[0]?.shape) {
    throw new Error('model input shape is missing')
  }
  let height = model.inputs[0].shape[1]
  let width = model.inputs[0].shape[2]
  if (!height || !width) {
    throw new Error('model input width/height is missing')
  }
  return { width, height }
}

async function loadImage(file: File) {
  let objectUrl = URL.createObjectURL(file)
  imageReady = false
  runButton.disabled = true
  try {
    await new Promise<void>((resolve, reject) => {
      image = new Image()
      image.onload = () => resolve()
      image.onerror = () => reject(new Error('failed to load image'))
      image.src = objectUrl
    })
    imageReady = true
    drawImageToCanvas()
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}

async function ensureModel() {
  let nextUrl = modelUrlInput.value.trim()
  if (!nextUrl) throw new Error('model path is required')
  if (model && nextUrl === modelUrl) return model

  setStatus('Loading model...')
  model?.dispose()
  model = await loadYoloModel(nextUrl)
  modelUrl = nextUrl

  let inputShape = getInputShape()
  canvas.width = inputShape.width
  canvas.height = inputShape.height
  drawImageToCanvas()
  showModelMetadata()
  setStatus(`Model ready (${inputShape.width}x${inputShape.height})`)
  runButton.disabled = !imageReady
  return model
}

function getNumClasses() {
  return parseOptionalNumber(numClassesInput) ?? model?.class_names?.length ?? 80
}

async function runClassify(startedAt: number) {
  let prediction = (
    await classifyImage({
      model: model!,
      pixels: canvas,
      num_classes: getNumClasses(),
    })
  )[0]

  let top = prediction.all_confidences
    .map((confidence, index) => ({
      confidence,
      label: model?.class_names?.[index] ?? String(index),
    }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 10)

  setResults(
    top.map(item => `${item.confidence.toFixed(4)} ${item.label}`),
  )
  setStatus(`classified in ${(performance.now() - startedAt).toFixed(1)} ms`)
  console.log('classify predictions:', prediction)
}

function drawDetectionBox(box: {
  x: number
  y: number
  width: number
  height: number
  confidence: number
  class_index: number
}) {
  let label = model?.class_names?.[box.class_index] ?? String(box.class_index)
  drawBox({
    context,
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
    borderColor: '#ef4444',
    lineWidth: 3,
    label: {
      text: `${box.confidence.toFixed(2)} ${label}`,
      fontColor: '#ffffff',
      backgroundColor: '#111827cc',
    },
  })
}

async function runDetect(startedAt: number) {
  let output_format = getOutputFormat() as BoxOutputFormat | undefined
  let predictions = await detectBox({
    tf,
    model: model!,
    pixels: canvas,
    num_classes: getNumClasses(),
    maxOutputSize: parseNumber(maxOutputSizeInput, 50),
    scoreThreshold: Number(scoreThresholdInput.value),
    iouThreshold: Number(iouThresholdInput.value),
    ...(output_format ? { output_format } : {}),
  })
  let boxes = predictions[0] ?? []
  boxes.forEach(drawDetectionBox)
  setResults(
    boxes.map(
      box =>
        `${box.confidence.toFixed(4)} ${model?.class_names?.[box.class_index] ?? box.class_index}`,
    ),
  )
  setStatus(`${boxes.length} boxes in ${(performance.now() - startedAt).toFixed(1)} ms`)
  console.log('detect predictions:', predictions)
}

async function runPose(startedAt: number) {
  let output_format = getOutputFormat() as PoseOutputFormat | undefined
  let predictions = await detectPose({
    tf,
    model: model!,
    pixels: canvas,
    num_classes: getNumClasses(),
    num_keypoints: parseOptionalNumber(numKeypointsInput) ?? model?.keypoints ?? 17,
    visibility: model?.visibility ?? true,
    maxOutputSize: parseNumber(maxOutputSizeInput, 50),
    scoreThreshold: Number(scoreThresholdInput.value),
    iouThreshold: Number(iouThresholdInput.value),
    ...(output_format ? { output_format } : {}),
  })
  let boxes = predictions[0] ?? []
  for (let box of boxes) {
    drawDetectionBox(box)
    for (let keypoint of box.keypoints) {
      drawBox({
        context,
        x: keypoint.x,
        y: keypoint.y,
        width: 4,
        height: 4,
        borderColor: '#22c55e',
        lineWidth: 2,
        label: {
          text: keypoint.visibility.toFixed(2),
          fontColor: '#ffffff',
          backgroundColor: '#166534cc',
          font: 'normal 700 11px Arial, sans-serif',
        },
      })
    }
  }
  setResults(
    boxes.map(box => `${box.confidence.toFixed(4)} ${box.keypoints.length} keypoints`),
  )
  setStatus(`${boxes.length} poses in ${(performance.now() - startedAt).toFixed(1)} ms`)
  console.log('pose predictions:', predictions)
}

function drawMaskOverlay(box: {
  x: number
  y: number
  width: number
  height: number
  class_index: number
  confidence: number
  all_confidences: number[]
  mask_coefficients: number[]
}, masks: number[][][]) {
  let boxRect = {
    left: box.x - box.width / 2,
    top: box.y - box.height / 2,
    right: box.x + box.width / 2,
    bottom: box.y + box.height / 2,
  }
  let mask = combineMask(box, masks)
  let maskHeight = mask.length
  let maskWidth = mask[0].length
  let tileWidth = canvas.width / maskWidth
  let tileHeight = canvas.height / maskHeight
  for (let y = 0; y < maskHeight; y++) {
    for (let x = 0; x < maskWidth; x++) {
      let left = (x / maskWidth) * canvas.width
      let top = (y / maskHeight) * canvas.height
      let rect = {
        left,
        top,
        right: left + tileWidth,
        bottom: top + tileHeight,
      }
      if (!hasOverlap(boxRect, rect)) continue
      context.fillStyle = `rgba(34,197,94,${mask[y][x] * 0.55})`
      context.fillRect(left, top, tileWidth, tileHeight)
    }
  }
}

async function runSegment(startedAt: number) {
  let output_format = getOutputFormat() as SegmentOutputFormat | undefined
  let predictions = await detectSegment({
    tf,
    model: model!,
    pixels: canvas,
    num_classes: getNumClasses(),
    num_channels: parseOptionalNumber(numChannelsInput) ?? 32,
    maxOutputSize: parseNumber(maxOutputSizeInput, 50),
    scoreThreshold: Number(scoreThresholdInput.value),
    iouThreshold: Number(iouThresholdInput.value),
    ...(output_format ? { output_format } : {}),
  })
  let result = predictions[0]
  let boxes = result?.bounding_boxes ?? []
  for (let box of boxes) {
    drawMaskOverlay(box, result.masks)
    drawDetectionBox(box)
  }
  setResults(
    boxes.map(
      box =>
        `${box.confidence.toFixed(4)} ${model?.class_names?.[box.class_index] ?? box.class_index}`,
    ),
  )
  setStatus(`${boxes.length} masks in ${(performance.now() - startedAt).toFixed(1)} ms`)
  console.log('segment predictions:', predictions)
}

async function runTask() {
  if (!imageReady) throw new Error('choose an image first')
  await ensureModel()
  drawImageToCanvas()
  setResults([])
  setStatus('Running...')
  runButton.disabled = true
  let startedAt = performance.now()
  let task = taskInput.value as Task
  if (task === 'classify') await runClassify(startedAt)
  if (task === 'detect') await runDetect(startedAt)
  if (task === 'pose') await runPose(startedAt)
  if (task === 'segment') await runSegment(startedAt)
  runButton.disabled = false
}

async function main() {
  if (location.protocol === 'file:') {
    throw new Error(
      'Open this page from npm run dev:yolo26:all, e.g. http://127.0.0.1:8000/test-all.html',
    )
  }

  loadButton.addEventListener('click', () => {
    ensureModel().catch(error => {
      console.error(error)
      setStatus(error instanceof Error ? error.message : String(error))
    })
  })

  fileInput.addEventListener('change', async () => {
    let file = fileInput.files?.[0]
    if (!file) return
    setStatus('Loading image...')
    await loadImage(file)
    runButton.disabled = !model
    setStatus(model ? 'Image ready' : 'Image ready; load a model next')
  })

  runButton.addEventListener('click', () => {
    runTask().catch(error => {
      console.error(error)
      setStatus(error instanceof Error ? error.message : String(error))
      runButton.disabled = false
    })
  })

  await ensureModel()
}

main().catch(error => {
  console.error(error)
  setStatus(error instanceof Error ? error.message : String(error))
})
