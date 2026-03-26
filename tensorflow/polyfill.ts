// Polyfill the removed util.isNullOrUndefined (Node 23+ broke tfjs-node < ~4.23)
export function polyfillUtil() {
  let util = eval('require("util")')

  // Only patch if the function is missing
  if (typeof util.isNullOrUndefined !== 'function') {
    util.isNullOrUndefined = (val: any): boolean =>
      val === null || val === undefined
  }

  // Optional: also patch isUndefined / isDefined if you see similar errors later
  if (typeof util.isUndefined !== 'function') {
    util.isUndefined = (val: any): boolean => val === undefined
  }

  if (typeof util.isDefined !== 'function') {
    util.isDefined = (val: any): boolean => val !== undefined
  }

  if (typeof util.isArray !== 'function') {
    util.isArray = (val: any): boolean => {
      return Array.isArray(val)
    }
  }
}

// only run in node.js
if (typeof window === 'undefined') {
  polyfillUtil()
}
