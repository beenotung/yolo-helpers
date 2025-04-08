import * as tf from '@tensorflow/tfjs'
import {
  ModelWithMetadata,
  detectBox,
  detectPose,
  loadTextFromUrl,
  loadYoloModel,
  ModelMetadata,
  parseMetadataYaml,
  BoundingBox,
  detectSegment,
  combineMask,
  hasOverlap,
} from '../../browser'

async function main() {
  let dom = {
    model_name: querySelector('#model-name'),
    status: querySelector('#status'),
    start_front_camera: querySelector(
      '#start-front-camera',
    ) as HTMLButtonElement,
    start_back_camera: querySelector('#start-back-camera') as HTMLButtonElement,
    stop_camera: querySelector('#stop-camera') as HTMLButtonElement,
    camera_container: querySelector('#camera-container'),
    camera_video: querySelector('#camera-video') as HTMLVideoElement,
    camera_canvas: querySelector('#camera-canvas') as HTMLCanvasElement,
  }
  let camera_context = dom.camera_canvas.getContext('2d')!
  let snapshot_canvas = document.createElement('canvas')
  let snapshot_context = snapshot_canvas.getContext('2d')!

  let model_url = ''
  let model: ModelWithMetadata<tf.GraphModel> | null = null

  dom.camera_canvas.onclick = () => {
    let canvas = dom.camera_canvas
    let scale = canvas.style.transform.includes('scaleX(-1)') ? 1 : -1
    canvas.style.transform = `scaleX(${scale})`
  }

  dom.model_name.onclick = async () => {
    let ans = prompt('model url:', model_url)
    if (ans) {
      model_url = ans
      if (model) {
        model.dispose()
      }
      model = await loadModel()
    }
  }

  // e.g. "saved_models/v11n200ep/model.json"
  model_url = await loadModelUrl()
  await loadModel()

  async function loadModel() {
    // release existing model if present
    if (model) {
      model.dispose()
    }

    let model_name = model_url.split('/').slice(-2)[0]
    dom.model_name.textContent = `(${model_name})`

    model = await loadYoloModel(model_url)
    if (!model.class_names) {
      throw new Error('class_names is not defined')
    }
    return model
  }

  let stream: MediaStream | null = null

  let timer = 0

  function stopCamera() {
    if (timer) {
      cancelAnimationFrame(timer)
      timer = 0
    }

    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      stream = null
      dom.camera_video.srcObject = null
    }
  }

  function calcSize() {
    let rect = dom.camera_container.getBoundingClientRect()
    let scale = Math.min(
      rect.width / dom.camera_video.videoWidth,
      rect.height / dom.camera_video.videoHeight,
    )
    scale = Math.min(scale, 1)
    let width = dom.camera_video.videoWidth * scale
    let height = dom.camera_video.videoHeight * scale

    dom.camera_video.width = width
    dom.camera_video.height = height

    dom.camera_canvas.width = width
    dom.camera_canvas.height = height

    snapshot_canvas.width = width
    snapshot_canvas.height = height

    let space = Math.max(0, (rect.width - width) / 2)
    dom.camera_video.style.left = `${space}px`
    dom.camera_canvas.style.left = `${space}px`

    console.log({ rect, scale, width, height })
  }

  window.onresize = calcSize

  async function startCamera(facingMode: 'environment' | 'user') {
    stopCamera()
    let video = dom.camera_video
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode },
    })
    let p = new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve
      video.onerror = reject
    })
    video.srcObject = stream
    video.play()
    await p

    calcSize()

    timer = requestAnimationFrame(tick)
  }

  let confidence_threshold = 0.5

  async function tick() {
    if (!model) {
      throw new Error('model is not loaded')
    }
    snapshot_context.drawImage(
      dom.camera_video,
      0,
      0,
      dom.camera_video.videoWidth,
      dom.camera_video.videoHeight,
      0,
      0,
      snapshot_canvas.width,
      snapshot_canvas.height,
    )

    function drawSnapshot() {
      camera_context.drawImage(
        snapshot_canvas,
        0,
        0,
        snapshot_canvas.width,
        snapshot_canvas.height,
        0,
        0,
        dom.camera_canvas.width,
        dom.camera_canvas.height,
      )
    }

    function drawBox(box: BoundingBox) {
      camera_context.strokeStyle = 'red'
      camera_context.lineWidth = 3
      camera_context.strokeRect(
        box.x - box.width / 2,
        box.y - box.height / 2,
        box.width,
        box.height,
      )
      let name = model!.class_names![box.class_index]
      camera_context.fillStyle = 'cyan'
      camera_context.font = '16px Arial'
      camera_context.fillText(
        name,
        box.x - box.width / 2,
        box.y - box.height / 2 - 8,
      )
    }

    if (model.task == 'detect') {
      let batches = await detectBox({
        model,
        tf,
        pixels: snapshot_canvas,
        num_classes: model.class_names!.length,
        scoreThreshold: confidence_threshold,
        maxOutputSize: 1,
      })
      drawSnapshot()
      for (let boxes of batches) {
        for (let box of boxes) {
          drawBox(box)
        }
      }
    }

    if (model.task == 'pose') {
      let batches = await detectPose({
        model,
        tf,
        pixels: snapshot_canvas,
        num_classes: model.class_names!.length,
        num_keypoints: model.keypoints!,
        visibility: model.visibility!,
        scoreThreshold: confidence_threshold,
        maxOutputSize: 1,
      })
      drawSnapshot()
      for (let boxes of batches) {
        for (let box of boxes) {
          drawBox(box)
          for (let keypoint of box.keypoints) {
            let width = 10
            let height = 10
            camera_context.strokeStyle = 'green'
            camera_context.lineWidth = 3
            camera_context.strokeRect(
              keypoint.x - width / 2,
              keypoint.y - height / 2,
              width,
              height,
            )
          }
        }
      }
    }

    if (model.task == 'segment') {
      let batches = await detectSegment({
        model,
        tf,
        pixels: snapshot_canvas,
        num_classes: model.class_names!.length,
        scoreThreshold: confidence_threshold,
        maxOutputSize: 1,
      })
      let canvas = dom.camera_canvas
      console.log({
        pix: snapshot_canvas.width,
        out: canvas.width,
      })
      drawSnapshot()
      for (let { bounding_boxes, masks } of batches) {
        for (let box of bounding_boxes) {
          drawBox(box)
          let boxRect = {
            left: box.x - box.width / 2,
            top: box.y - box.height / 2,
            right: box.x + box.width / 2,
            bottom: box.y + box.height / 2,
          }
          let mask = combineMask(box, masks)
          let H = mask.length
          let W = mask[0].length
          for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
              let width = canvas.width / W
              let height = canvas.height / H

              let left = (w / W) * canvas.width
              let top = (h / H) * canvas.height

              let right = left + width
              let bottom = top + height

              let maskRect = { left, top, right, bottom }

              if (!hasOverlap(boxRect, maskRect)) {
                // continue
              }

              let v = mask[h][w]
              camera_context.fillStyle = `rgba(0,255,0,${v})`
              camera_context.fillRect(left, top, width, height)
            }
          }
        }
      }
    }

    timer = requestAnimationFrame(tick)
  }

  dom.start_front_camera.onclick = () => {
    startCamera('user')
    dom.start_front_camera.disabled = true
    dom.start_back_camera.disabled = false
    dom.stop_camera.disabled = false
  }
  dom.start_front_camera.disabled = false

  dom.start_back_camera.onclick = () => {
    startCamera('environment')
    dom.start_front_camera.disabled = false
    dom.start_back_camera.disabled = true
    dom.stop_camera.disabled = false
  }
  dom.start_back_camera.disabled = false

  dom.stop_camera.onclick = () => {
    stopCamera()
    dom.start_front_camera.disabled = false
    dom.start_back_camera.disabled = false
    dom.stop_camera.disabled = true
  }

  dom.status.textContent = 'Ready'
}

async function loadModelUrl(url: string = 'model-url.txt') {
  let text = await loadTextFromUrl(url)
  let lines = text.split('\n')
  for (let line of lines) {
    line = line.trim()
    // skip empty lines
    if (line.length == 0) continue
    // skip comments starting with '#'
    if (line.startsWith('#')) continue
    // return the first valid line
    return line
  }
  throw new Error('No model URL found')
}

function querySelector(selector: string, parent: HTMLElement = document.body) {
  let node = parent.querySelector<HTMLElement>(selector)
  if (!node) {
    throw new Error(`querySelector: ${selector} not found`)
  }
  return node
}

main().catch(e => {
  console.error(e)
  alert(String(e))
})
