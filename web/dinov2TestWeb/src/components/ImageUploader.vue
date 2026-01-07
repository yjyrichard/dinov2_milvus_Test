<script setup>
import { ref } from 'vue'

const emit = defineEmits(['upload'])

const isDragging = ref(false)
const previewUrl = ref(null)
const selectedFile = ref(null)

function handleDragOver(e) {
  e.preventDefault()
  isDragging.value = true
}

function handleDragLeave() {
  isDragging.value = false
}

function handleDrop(e) {
  e.preventDefault()
  isDragging.value = false

  const files = e.dataTransfer.files
  if (files.length > 0) {
    handleFile(files[0])
  }
}

function handleFileSelect(e) {
  const files = e.target.files
  if (files.length > 0) {
    handleFile(files[0])
  }
}

function handleFile(file) {
  // 验证文件类型
  const validTypes = ['image/tiff', 'image/jpeg', 'image/png', 'image/gif']
  if (!validTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.tif')) {
    alert('请上传 TIF、JPG、PNG 格式的图片')
    return
  }

  selectedFile.value = file

  // 创建预览
  const reader = new FileReader()
  reader.onload = (e) => {
    previewUrl.value = e.target.result
  }
  reader.readAsDataURL(file)

  // 触发搜索
  emit('upload', file)
}

function clearSelection() {
  selectedFile.value = null
  previewUrl.value = null
}
</script>

<template>
  <div class="uploader-container">
    <!-- 上传区域 -->
    <div
      class="upload-zone"
      :class="{ dragging: isDragging, 'has-preview': previewUrl }"
      @dragover="handleDragOver"
      @dragleave="handleDragLeave"
      @drop="handleDrop"
      @click="$refs.fileInput.click()"
    >
      <input
        ref="fileInput"
        type="file"
        accept=".tif,.tiff,.jpg,.jpeg,.png"
        style="display: none"
        @change="handleFileSelect"
      />

      <template v-if="!previewUrl">
        <div class="upload-icon">
          <svg viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
            <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
          </svg>
        </div>
        <p class="upload-text">拖拽图片到此处，或点击上传</p>
        <p class="upload-hint">支持 TIF / JPG / PNG</p>
      </template>

      <template v-else>
        <div class="preview-container">
          <img :src="previewUrl" alt="Preview" class="preview-image" />
          <button class="clear-btn" @click.stop="clearSelection">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
          </button>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.uploader-container {
  width: 100%;
}

.upload-zone {
  border: 2px dashed #ccc;
  border-radius: 8px;
  padding: 40px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: #fafafa;
}

.upload-zone:hover,
.upload-zone.dragging {
  border-color: #409eff;
  background: #ecf5ff;
}

.upload-zone.has-preview {
  padding: 10px;
}

.upload-icon {
  color: #c0c4cc;
  margin-bottom: 16px;
}

.upload-text {
  font-size: 16px;
  color: #606266;
  margin: 0 0 8px 0;
}

.upload-hint {
  font-size: 12px;
  color: #909399;
  margin: 0;
}

.preview-container {
  position: relative;
  display: inline-block;
}

.preview-image {
  max-width: 100%;
  max-height: 200px;
  border-radius: 4px;
}

.clear-btn {
  position: absolute;
  top: -10px;
  right: -10px;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: none;
  background: #f56c6c;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
}

.clear-btn:hover {
  background: #f78989;
}
</style>
