import * as tf from '@tensorflow/tfjs'
import { detectBox, drawBox, loadYoloModel } from '../browser'

const modelUrl = 'models/yolo26n_web_model'
const maxOutputSize = 50

const fileInput = document.querySelector<HTMLInputElement>('#file')!
const scoreThresholdInput =
  document.querySelector<HTMLInputElement>('#scoreThreshold')!
const iouThresholdInput =
  document.querySelector<HTMLInputElement>('#iouThreshold')!
const runButton = document.querySelector<HTMLButtonElement>('#run')!
const status = document.querySelector<HTMLDivElement>('#status')!
const canvas = document.querySelector<HTMLCanvasElement>('#canvas')!
const context = canvas.getContext('2d')!

let image = new Image()
let imageReady = false

function setStatus(text: string) {
  status.textContent = text
}

function parseThreshold(input: HTMLInputElement, fallback: number) {
  let value = Number(input.value)
  if (Number.isFinite(value)) return value
  return fallback
}

function drawImageToCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height)
  context.drawImage(image, 0, 0, canvas.width, canvas.height)
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

async function main() {
  if (location.protocol === 'file:') {
    throw new Error(
      'Open this page from npm run dev:yolo26, e.g. http://127.0.0.1:8000/test.html',
    )
  }

  setStatus('Loading model...')
  const model = await loadYoloModel(modelUrl)
  console.log({model})
  debugger
  const inputHeight = model.inputs[0].shape![1]
  const inputWidth = model.inputs[0].shape![2]
  canvas.width = inputWidth
  canvas.height = inputHeight

  setStatus(`Model ready (${inputWidth}x${inputHeight})`)
  runButton.disabled = !imageReady

  fileInput.addEventListener('change', async () => {
    let file = fileInput.files?.[0]
    if (!file) return
    setStatus('Loading image...')
    await loadImage(file)
    runButton.disabled = false
    setStatus('Image ready')
  })

  runButton.addEventListener('click', async () => {
    if (!imageReady) return

    let scoreThreshold = parseThreshold(scoreThresholdInput, 0.25)
    let iouThreshold = parseThreshold(iouThresholdInput, 0.45)

    drawImageToCanvas()
    setStatus('Detecting...')
    runButton.disabled = true

    let startedAt = performance.now()
    let predictions = await detectBox({
      tf,
      model,
      pixels: canvas,
      output_format: 'yolo26',
      num_classes: model.class_names?.length ?? 80,
      maxOutputSize,
      scoreThreshold,
      iouThreshold,
    })
    let elapsed = performance.now() - startedAt

    console.log('predictions:', predictions)

    let boxes = predictions[0] ?? []
    for (let box of boxes) {
      let name = model.class_names?.[box.class_index] ?? String(box.class_index)
      drawBox({
        context,
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
        borderColor: '#ef4444',
        lineWidth: 3,
        label: {
          text: `${box.confidence.toFixed(2)} ${name}`,
          fontColor: '#ffffff',
          backgroundColor: '#111827cc',
        },
      })
    }

    runButton.disabled = false
    setStatus(`${boxes.length} boxes in ${elapsed.toFixed(1)} ms`)
  })
}

main().catch(error => {
  console.error(error)
  setStatus(error instanceof Error ? error.message : String(error))
})
