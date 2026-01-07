<script setup>
import PatentGroup from './PatentGroup.vue'

const props = defineProps({
  results: {
    type: Array,
    default: () => []
  },
  queryInfo: {
    type: Object,
    default: () => ({})
  },
  loading: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['view-detail'])

function handleViewDetail(data) {
  emit('view-detail', data)
}
</script>

<template>
  <div class="search-results">
    <div class="results-header">
      <h3 class="results-title">搜索结果</h3>
      <span v-if="queryInfo.total_matched !== undefined" class="results-count">
        共 {{ queryInfo.total_matched }} 个匹配
      </span>
    </div>

    <div v-if="loading" class="loading-state">
      <div class="loading-spinner"></div>
      <p>搜索中...</p>
    </div>

    <div v-else-if="results.length === 0" class="empty-state">
      <svg viewBox="0 0 24 24" width="48" height="48" fill="#c0c4cc">
        <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
      </svg>
      <p>上传图片开始搜索</p>
    </div>

    <div v-else class="results-list">
      <PatentGroup
        v-for="group in results"
        :key="group.patent_id"
        :patent-id="group.patent_id"
        :pages="group.pages"
        :max-score="group.max_score"
        @view-detail="handleViewDetail"
      />
    </div>
  </div>
</template>

<style scoped>
.search-results {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 16px;
  min-height: 300px;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.results-title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.results-count {
  font-size: 14px;
  color: #909399;
}

.loading-state,
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  color: #909399;
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #ebeef5;
  border-top-color: #409eff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.loading-state p,
.empty-state p {
  margin: 16px 0 0 0;
  font-size: 14px;
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
