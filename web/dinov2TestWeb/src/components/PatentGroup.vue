<script setup>
import { getThumbnailUrl, getFullImageUrl } from '../api'

const props = defineProps({
  patentId: {
    type: String,
    required: true
  },
  pages: {
    type: Array,
    required: true
  },
  maxScore: {
    type: Number,
    required: true
  }
})

const emit = defineEmits(['view-detail'])

function viewDetail(page) {
  emit('view-detail', {
    ...page,
    patent_id: props.patentId,
    fullImageUrl: getFullImageUrl(page.file_name)
  })
}

function getScoreColor(score) {
  if (score >= 0.9) return '#67c23a'
  if (score >= 0.7) return '#409eff'
  if (score >= 0.5) return '#e6a23c'
  return '#909399'
}
</script>

<template>
  <div class="patent-group">
    <div class="group-header">
      <span class="patent-id">{{ patentId }}</span>
      <span class="max-score" :style="{ color: getScoreColor(maxScore) }">
        最高相似度: {{ (maxScore * 100).toFixed(1) }}%
      </span>
    </div>

    <div class="pages-grid">
      <div
        v-for="page in pages"
        :key="page.id"
        class="page-card"
        @click="viewDetail(page)"
      >
        <div class="thumbnail-container">
          <img
            :src="getThumbnailUrl(page.file_name)"
            :alt="page.file_name"
            loading="lazy"
            class="thumbnail"
          />
        </div>
        <div class="page-info">
          <span class="page-num">{{ page.page_num }}</span>
          <span class="score" :style="{ color: getScoreColor(page.score) }">
            {{ (page.score * 100).toFixed(1) }}%
          </span>
        </div>
        <div class="score-bar-container">
          <div
            class="score-bar"
            :style="{
              width: (page.score * 100) + '%',
              backgroundColor: getScoreColor(page.score)
            }"
          ></div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.patent-group {
  background: #fff;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #ebeef5;
}

.patent-id {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.max-score {
  font-size: 14px;
  font-weight: 500;
}

.pages-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 12px;
}

.page-card {
  cursor: pointer;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid #ebeef5;
  transition: all 0.3s ease;
}

.page-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  border-color: #409eff;
}

.thumbnail-container {
  aspect-ratio: 1;
  overflow: hidden;
  background: #f5f7fa;
}

.thumbnail {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.page-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
  font-size: 12px;
}

.page-num {
  color: #606266;
}

.score {
  font-weight: 600;
}

.score-bar-container {
  height: 3px;
  background: #ebeef5;
}

.score-bar {
  height: 100%;
  transition: width 0.3s ease;
}
</style>
