<script setup>
import { ref, onMounted } from 'vue'
import ImageUploader from './components/ImageUploader.vue'
import TimingPanel from './components/TimingPanel.vue'
import SearchResults from './components/SearchResults.vue'
import BatchImportPanel from './components/BatchImportPanel.vue'
import ImageDetailModal from './components/ImageDetailModal.vue'
import { searchImage, getCollectionStats } from './api'

// 状态
const loading = ref(false)
const results = ref([])
const timing = ref({})
const queryInfo = ref({})
const collectionStats = ref(null)
const error = ref(null)

// 弹窗状态
const modalVisible = ref(false)
const selectedImage = ref(null)

// 搜索参数
const topK = ref(10)
const minScore = ref(0.4)

// 获取统计信息
async function fetchStats() {
  try {
    const res = await getCollectionStats()
    collectionStats.value = res.data
  } catch (e) {
    console.error('Failed to fetch stats:', e)
  }
}

// 处理上传搜索
async function handleUpload(file) {
  loading.value = true
  error.value = null

  try {
    const res = await searchImage(file, topK.value, minScore.value)

    if (res.code === 0) {
      results.value = res.data.results
      timing.value = res.data.timing
      queryInfo.value = res.data.query_info
    } else {
      error.value = res.message
    }
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

// 查看详情
function handleViewDetail(data) {
  selectedImage.value = data
  modalVisible.value = true
}

// 关闭弹窗
function handleCloseModal() {
  modalVisible.value = false
  selectedImage.value = null
}

onMounted(() => {
  fetchStats()
})
</script>

<template>
  <div class="app">
    <header class="header">
      <div class="header-content">
        <h1 class="title">DINOv2 图像检索系统</h1>
        <div v-if="collectionStats" class="db-status">
          <span class="status-dot"></span>
          <span>数据库: {{ collectionStats.collection?.num_entities || 0 }} 张图片</span>
        </div>
      </div>
    </header>

    <main class="main">
      <div class="container">
        <!-- 上传区域 -->
        <section class="upload-section">
          <ImageUploader @upload="handleUpload" />
        </section>

        <!-- 中间区域：查询预览 + 性能面板 -->
        <section class="middle-section">
          <div class="timing-wrapper">
            <TimingPanel :timing="timing" />
          </div>
          <div class="import-wrapper">
            <BatchImportPanel />
          </div>
        </section>

        <!-- 错误提示 -->
        <div v-if="error" class="error-alert">
          <span>{{ error }}</span>
          <button @click="error = null">关闭</button>
        </div>

        <!-- 搜索结果 -->
        <section class="results-section">
          <SearchResults
            :results="results"
            :query-info="queryInfo"
            :loading="loading"
            @view-detail="handleViewDetail"
          />
        </section>
      </div>
    </main>

    <!-- 详情弹窗 -->
    <ImageDetailModal
      :visible="modalVisible"
      :image-data="selectedImage"
      @close="handleCloseModal"
    />
  </div>
</template>

<style>
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: #f5f7fa;
  color: #303133;
}

#app {
  min-height: 100vh;
}
</style>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  background: #fff;
  border-bottom: 1px solid #ebeef5;
  padding: 16px 24px;
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.title {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  color: #303133;
}

.db-status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  color: #606266;
}

.status-dot {
  width: 8px;
  height: 8px;
  background: #67c23a;
  border-radius: 50%;
}

.main {
  flex: 1;
  padding: 24px;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.upload-section {
  margin-bottom: 24px;
}

.middle-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

@media (max-width: 768px) {
  .middle-section {
    grid-template-columns: 1fr;
  }
}

.timing-wrapper,
.import-wrapper {
  min-width: 0;
}

.error-alert {
  background: #fef0f0;
  border: 1px solid #fbc4c4;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #f56c6c;
}

.error-alert button {
  background: none;
  border: none;
  color: #f56c6c;
  cursor: pointer;
  font-size: 12px;
}

.results-section {
  margin-bottom: 24px;
}
</style>
