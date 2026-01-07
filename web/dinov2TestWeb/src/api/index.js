/**
 * API 模块
 */

// 自动检测 API 地址
// 如果通过 Vite dev server (3000端口) 访问，使用代理 /api
// 如果直接访问后端 (8000端口)，也使用 /api
const API_BASE = '/api'

/**
 * 通用请求封装
 */
async function request(url, options = {}) {
  console.log(`[API] ${options.method || 'GET'} ${url}`)

  try {
    const response = await fetch(url, options)

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`[API] Error: ${response.status} ${response.statusText}`, errorText)
      throw new Error(`Request failed: ${response.statusText}`)
    }

    const data = await response.json()
    console.log(`[API] Response:`, data)
    return data
  } catch (error) {
    console.error(`[API] Request failed:`, error)
    throw error
  }
}

/**
 * 搜索图片
 */
export async function searchImage(file, topK = 10, minScore = 0.4) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('top_k', topK)
  formData.append('min_score', minScore)

  return request(`${API_BASE}/search`, {
    method: 'POST',
    body: formData
  })
}

/**
 * 获取 Collection 统计
 */
export async function getCollectionStats() {
  return request(`${API_BASE}/collection/stats`)
}

/**
 * 启动批量导入
 */
export async function startBatchImport() {
  return request(`${API_BASE}/batch/start`, {
    method: 'POST'
  })
}

/**
 * 获取导入状态
 */
export async function getBatchStatus() {
  return request(`${API_BASE}/batch/status`)
}

/**
 * 重置导入状态
 */
export async function resetBatchStatus() {
  return request(`${API_BASE}/batch/reset`, {
    method: 'POST'
  })
}

/**
 * 获取缩略图 URL
 */
export function getThumbnailUrl(fileName) {
  return `${API_BASE}/image/${fileName}`
}

/**
 * 获取原图 URL
 */
export function getFullImageUrl(fileName) {
  return `${API_BASE}/image/full/${fileName}`
}
