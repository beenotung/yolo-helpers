export type DrawBoxArgs = {
  /** canvas context to draw on */
  context: CanvasRenderingContext2D

  /** x-axis of the center of the box, in pixel unit */
  x: number

  /** y-axis of the center of the box, in pixel unit */
  y: number

  /** width of the box, in pixel unit */
  width: number

  /** height of the box, in pixel unit */
  height: number

  /** color of the border of the box, default is `red` */
  borderColor?: string

  /** line width of the box, in pixel unit, default is 5px */
  lineWidth?: number

  /** label of the box, e.g. class name, confidence score, etc. */
  label?: {
    text: string
    /** color of the text label, default is `'white'` */
    fontColor?: string
    /** background color of the text label, default is `'transparent'` */
    backgroundColor?: string
    /** font style of the text label, default is `'normal 900 14px Arial, sans-serif'` */
    font?: string
  }
}

export function drawBox(args: DrawBoxArgs) {
  let { context, label } = args

  let lineWidth = args.lineWidth ?? 5
  let borderColor = args.borderColor ?? 'red'

  let left = args.x - args.width / 2
  let top = args.y - args.height / 2

  context.lineWidth = lineWidth
  context.strokeStyle = borderColor
  context.strokeRect(left, top, args.width, args.height)

  if (label) {
    let textColor = label.fontColor ?? 'white'
    let backgroundColor = label.backgroundColor ?? 'transparent'
    let font = label.font ?? 'normal 900 14px Arial, sans-serif'
    let text = label.text

    context.font = font
    let metrics = context.measureText(text)
    let width = metrics.width
    let height = metrics.fontBoundingBoxAscent || metrics.actualBoundingBoxAscent || 25

    if (top - lineWidth - height >= 0) {
      // draw background of text label
      if (backgroundColor !== 'transparent') {
        context.fillStyle = backgroundColor
        context.fillRect(
          left,
          top - lineWidth - height,
          width,
          height + lineWidth,
        )
      }
    } else {
      top += height + lineWidth
      left += lineWidth
      // draw background of text label
      if (backgroundColor !== 'transparent') {
        context.fillStyle = backgroundColor
        context.fillRect(
          left,
          top - lineWidth - height,
          width,
          height + lineWidth,
        )
      }
    }

    // draw the text label
    context.fillStyle = textColor
    context.fillText(text, left, top - lineWidth)
  }
}
