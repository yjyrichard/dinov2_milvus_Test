<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { startBatchImport, getBatchStatus, resetBatchStatus } from '../api'

const status = ref(null)
const loading = ref(false)
const error = ref(null)
let pollInterval = null

async function fetchStatus() {
  try {
    const res = await getBatchStatus()
    status.value = res.data
    error.value = null
  } catch (e) {
    error.value = e.message
  }
}

async function handleStart() {
  loading.value = true
  error.value = null

  try {
    await startBatchImport()
    startPolling()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

async function handleReset() {
  try {
    await resetBatchStatus()
    await fetchStatus()
  } catch (e) {
    error.value = e.message
  }
}

function startPolling() {
  if (pollInterval) return

  pollInterval = setInterval(async () => {
    await fetchStatus()

    // 如果完成或错误，停止轮询
    if (status.value?.status === 'completed' || status.value?.status === 'error') {
      stopPolling()
    }
  }, 1000)
}

function stopPolling() {
  if (pollInterval) {
    clearInterval(pollInterval)
    pollInterval = null
  }
}

function getStatusColor(s) {
  const colors = {
    idle: '#909399',
    running: '#409eff',
    completed: '#67c23a',
    error: '#f56c6c'
  }
  return colors[s] || '#909399'
}

function getStatusText(s) {
  const texts = {
    idle: '空闲',
    running: '运行中',
    completed: '已完成',
    error: '错误'
  }
  return texts[s] || s
}

onMounted(() => {
  fetchStatus()

  // 如果正在运行，开始轮询
  if (status.value?.status === 'running') {
    startPolling()
  }
})

onUnmounted(() => {
  stopPolling()
})
</script>

<template>
  <div class="batch-panel">
    <div class="panel-header">
      <h3 class="panel-title">批量导入</h3>
      <span
        class="status-badge"
        :style="{ backgroundColor: getStatusColor(status?.status) }"
      >
        {{ getStatusText(status?.status) }}
      </span>
    </div>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <div v-if="status" class="status-info">
      <div class="progress-section">
        <div class="progress-header">
          <span>进度: {{ status.processed }} / {{ status.total }}</span>
          <span>{{ status.progress_percent }}%</span>
        </div>
        <div class="progress-bar-container">
          <div
            class="progress-bar"
            :style="{ width: status.progress_percent + '%' }"
          ></div>
        </div>
      </div>

      <div class="stats-grid">
        <div class="stat-item">
          <span class="stat-label">速度</span>
          <span class="stat-value">{{ status.avg_speed }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">已用时</span>
          <span class="stat-value">{{ status.elapsed_sec }}s</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">预计剩余</span>
          <span class="stat-value">{{ status.estimated_remaining_sec }}s</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">失败</span>
          <span class="stat-value" :class="{ 'has-error': status.failed > 0 }">
            {{ status.failed }}
          </span>
        </div>
      </div>

      <div v-if="status.failed_files?.length > 0" class="failed-files">
        <p class="failed-title">失败文件:</p>
        <ul>
          <li v-for="file in status.failed_files" :key="file">{{ file }}</li>
        </ul>
      </div>
    </div>

    <div class="actions">
      <button
        class="btn btn-primary"
        :disabled="loading || status?.status === 'running'"
        @click="handleStart"
      >
        {{ loading ? '启动中...' : '开始导入' }}
      </button>
      <button
        class="btn btn-secondary"
        :disabled="status?.status === 'running'"
        @click="handleReset"
      >
        重置状态
      </button>
    </div>
  </div>
</template>

<style scoped>
.batch-panel {
  background: #fff;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.panel-title {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}

.status-badge {
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 12px;
  color: white;
}

.error-message {
  padding: 8px 12px;
  background: #fef0f0;
  border-radius: 4px;
  color: #f56c6c;
  font-size: 12px;
  margin-bottom: 12px;
}

.status-info {
  margin-bottom: 16px;
}

.progress-section {
  margin-bottom: 16px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #606266;
  margin-bottom: 6px;
}

.progress-bar-container {
  height: 8px;
  background: #ebeef5;
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: #409eff;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stat-label {
  font-size: 11px;
  color: #909399;
}

.stat-value {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
}

.stat-value.has-error {
  color: #f56c6c;
}

.failed-files {
  margin-top: 12px;
  padding: 8px;
  background: #fef0f0;
  border-radius: 4px;
  font-size: 12px;
}

.failed-title {
  margin: 0 0 4px 0;
  color: #f56c6c;
}

.failed-files ul {
  margin: 0;
  padding-left: 16px;
  color: #606266;
}

.actions {
  display: flex;
  gap: 8px;
}

.btn {
  flex: 1;
  padding: 8px 16px;
  border-radius: 4px;
  border: none;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary {
  background: #409eff;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #66b1ff;
}

.btn-secondary {
  background: #f5f7fa;
  color: #606266;
  border: 1px solid #dcdfe6;
}

.btn-secondary:hover:not(:disabled) {
  background: #ecf5ff;
  color: #409eff;
  border-color: #c6e2ff;
}
</style>
