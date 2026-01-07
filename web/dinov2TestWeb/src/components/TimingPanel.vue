<script setup>
import { computed } from 'vue'

const props = defineProps({
  timing: {
    type: Object,
    default: () => ({})
  }
})

const hasData = computed(() => {
  return props.timing && props.timing.total_ms
})

const maxTime = computed(() => {
  if (!hasData.value) return 100
  return Math.max(
    props.timing.feature_extraction_ms || 0,
    props.timing.milvus_search_ms || 0,
    props.timing.post_process_ms || 0
  )
})

function getBarWidth(value) {
  if (!maxTime.value || !value) return 0
  return (value / maxTime.value) * 100
}

function getBarColor(type) {
  const colors = {
    feature: '#409eff',
    search: '#67c23a',
    post: '#e6a23c',
    total: '#909399'
  }
  return colors[type] || '#409eff'
}
</script>

<template>
  <div class="timing-panel" :class="{ 'has-data': hasData }">
    <h3 class="panel-title">性能指标</h3>

    <div v-if="hasData" class="timing-items">
      <div class="timing-item">
        <div class="timing-header">
          <span class="timing-label">特征提取</span>
          <span class="timing-value">{{ timing.feature_extraction_ms?.toFixed(1) }}ms</span>
        </div>
        <div class="timing-bar-container">
          <div
            class="timing-bar"
            :style="{
              width: getBarWidth(timing.feature_extraction_ms) + '%',
              backgroundColor: getBarColor('feature')
            }"
          ></div>
        </div>
      </div>

      <div class="timing-item">
        <div class="timing-header">
          <span class="timing-label">向量检索</span>
          <span class="timing-value">{{ timing.milvus_search_ms?.toFixed(1) }}ms</span>
        </div>
        <div class="timing-bar-container">
          <div
            class="timing-bar"
            :style="{
              width: getBarWidth(timing.milvus_search_ms) + '%',
              backgroundColor: getBarColor('search')
            }"
          ></div>
        </div>
      </div>

      <div class="timing-item">
        <div class="timing-header">
          <span class="timing-label">后处理</span>
          <span class="timing-value">{{ timing.post_process_ms?.toFixed(1) }}ms</span>
        </div>
        <div class="timing-bar-container">
          <div
            class="timing-bar"
            :style="{
              width: getBarWidth(timing.post_process_ms) + '%',
              backgroundColor: getBarColor('post')
            }"
          ></div>
        </div>
      </div>

      <div class="timing-item total">
        <div class="timing-header">
          <span class="timing-label">总耗时</span>
          <span class="timing-value total-value">{{ timing.total_ms?.toFixed(1) }}ms</span>
        </div>
      </div>
    </div>

    <div v-else class="no-data">
      <p>上传图片后显示性能指标</p>
    </div>
  </div>
</template>

<style scoped>
.timing-panel {
  background: #fff;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.panel-title {
  margin: 0 0 16px 0;
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}

.timing-items {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.timing-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.timing-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.timing-label {
  font-size: 12px;
  color: #606266;
}

.timing-value {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
  font-family: monospace;
}

.timing-bar-container {
  height: 6px;
  background: #ebeef5;
  border-radius: 3px;
  overflow: hidden;
}

.timing-bar {
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s ease;
}

.timing-item.total {
  margin-top: 8px;
  padding-top: 12px;
  border-top: 1px solid #ebeef5;
}

.total-value {
  font-size: 18px;
  color: #409eff;
}

.no-data {
  text-align: center;
  padding: 20px;
}

.no-data p {
  margin: 0;
  color: #909399;
  font-size: 12px;
}
</style>
