<script setup>
import { ref } from 'vue'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  imageData: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['close'])

const imageLoading = ref(true)

function handleClose() {
  emit('close')
}

function handleImageLoad() {
  imageLoading.value = false
}
</script>

<template>
  <Teleport to="body">
    <div v-if="visible" class="modal-overlay" @click.self="handleClose">
      <div class="modal-content">
        <button class="close-btn" @click="handleClose">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>

        <div v-if="imageData" class="detail-container">
          <div class="image-section">
            <div v-if="imageLoading" class="loading-placeholder">
              <div class="loading-spinner"></div>
            </div>
            <img
              :src="imageData.fullImageUrl"
              :alt="imageData.file_name"
              class="full-image"
              @load="handleImageLoad"
            />
          </div>

          <div class="info-section">
            <h3 class="info-title">图片信息</h3>

            <div class="info-item">
              <span class="info-label">专利号</span>
              <span class="info-value">{{ imageData.patent_id }}</span>
            </div>

            <div class="info-item">
              <span class="info-label">页码</span>
              <span class="info-value">{{ imageData.page_num }}</span>
            </div>

            <div class="info-item">
              <span class="info-label">文件名</span>
              <span class="info-value filename">{{ imageData.file_name }}</span>
            </div>

            <div class="info-item">
              <span class="info-label">相似度</span>
              <span class="info-value score">
                {{ (imageData.score * 100).toFixed(1) }}%
              </span>
            </div>

            <div class="score-visual">
              <div
                class="score-fill"
                :style="{ width: (imageData.score * 100) + '%' }"
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
}

.modal-content {
  background: #fff;
  border-radius: 12px;
  max-width: 900px;
  max-height: 90vh;
  width: 100%;
  overflow: hidden;
  position: relative;
  display: flex;
  flex-direction: column;
}

.close-btn {
  position: absolute;
  top: 12px;
  right: 12px;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: none;
  background: rgba(0, 0, 0, 0.5);
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  transition: background 0.3s ease;
}

.close-btn:hover {
  background: rgba(0, 0, 0, 0.7);
}

.detail-container {
  display: flex;
  flex-direction: column;
}

@media (min-width: 768px) {
  .detail-container {
    flex-direction: row;
  }
}

.image-section {
  flex: 1;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  position: relative;
}

.loading-placeholder {
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.full-image {
  max-width: 100%;
  max-height: 70vh;
  object-fit: contain;
}

.info-section {
  padding: 24px;
  min-width: 250px;
}

.info-title {
  margin: 0 0 20px 0;
  font-size: 18px;
  font-weight: 600;
  color: #303133;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 16px;
}

.info-label {
  font-size: 12px;
  color: #909399;
}

.info-value {
  font-size: 16px;
  color: #303133;
  font-weight: 500;
}

.info-value.filename {
  font-size: 12px;
  font-family: monospace;
  word-break: break-all;
}

.info-value.score {
  font-size: 24px;
  color: #409eff;
}

.score-visual {
  height: 8px;
  background: #ebeef5;
  border-radius: 4px;
  overflow: hidden;
  margin-top: 8px;
}

.score-fill {
  height: 100%;
  background: linear-gradient(90deg, #409eff, #67c23a);
  border-radius: 4px;
  transition: width 0.3s ease;
}
</style>
