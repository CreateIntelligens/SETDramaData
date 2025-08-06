#!/bin/bash

# 時間統計日誌工具
# 記錄每個步驟的開始和結束時間

# 日誌檔案路徑
TIMING_LOG_FILE="${TIMING_LOG_FILE:-processing_timing.txt}"

# 記錄步驟開始時間
log_step_start() {
    local episode_num="$1"
    local step_name="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] 第${episode_num}集 - ${step_name} - 開始" >> "$TIMING_LOG_FILE"
}

# 記錄步驟結束時間
log_step_end() {
    local episode_num="$1"
    local step_name="$2"
    local start_time="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_formatted=$(printf "%02d:%02d:%02d" $((duration/3600)) $((duration%3600/60)) $((duration%60)))
    
    echo "[$timestamp] 第${episode_num}集 - ${step_name} - 完成 (耗時: ${duration_formatted})" >> "$TIMING_LOG_FILE"
}

# 記錄步驟失敗
log_step_failed() {
    local episode_num="$1"
    local step_name="$2"
    local start_time="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_formatted=$(printf "%02d:%02d:%02d" $((duration/3600)) $((duration%3600/60)) $((duration%60)))
    
    echo "[$timestamp] 第${episode_num}集 - ${step_name} - 失敗 (耗時: ${duration_formatted})" >> "$TIMING_LOG_FILE"
}

# 初始化日誌檔案（如果不存在）
init_timing_log() {
    if [ ! -f "$TIMING_LOG_FILE" ]; then
        echo "#  TTS 處理時間統計日誌" > "$TIMING_LOG_FILE"
        echo "# 格式: [時間戳記] 第X集 - 步驟名稱 - 狀態 (耗時: HH:MM:SS)" >> "$TIMING_LOG_FILE"
        echo "# =================================================" >> "$TIMING_LOG_FILE"
        echo "" >> "$TIMING_LOG_FILE"
    fi
}

# 顯示統計摘要
show_timing_summary() {
    local episode_num="$1"
    
    if [ ! -f "$TIMING_LOG_FILE" ]; then
        echo "❌ 時間統計檔案不存在"
        return 1
    fi
    
    echo ""
    echo "📊 第 $episode_num 集處理時間統計："
    echo "================================"
    
    # 提取該集的統計資料
    grep "第${episode_num}集" "$TIMING_LOG_FILE" | grep "完成\|失敗" | while read -r line; do
        # 解析時間和步驟資訊
        if [[ "$line" =~ \[([^]]+)\].*第${episode_num}集.*-\ ([^-]+)\ -\ ([^(]+).*耗時:\ ([^)]+) ]]; then
            local timestamp="${BASH_REMATCH[1]}"
            local step="${BASH_REMATCH[2]// /}"
            local status="${BASH_REMATCH[3]// /}"
            local duration="${BASH_REMATCH[4]}"
            
            local status_icon="✅"
            if [[ "$status" == "失敗" ]]; then
                status_icon="❌"
            fi
            
            echo "  $status_icon $step: $duration"
        fi
    done
    
    echo ""
}

# 顯示所有集數的統計
show_all_timing_summary() {
    if [ ! -f "$TIMING_LOG_FILE" ]; then
        echo "❌ 時間統計檔案不存在"
        return 1
    fi
    
    echo ""
    echo "📊 所有集數處理時間統計："
    echo "========================"
    
    # 提取所有集數
    local episodes=$(grep -o "第[0-9]\+集" "$TIMING_LOG_FILE" | sort -u | sed 's/第\([0-9]\+\)集/\1/')
    
    for episode in $episodes; do
        echo ""
        echo "第 $episode 集："
        echo "--------"
        
        grep "第${episode}集" "$TIMING_LOG_FILE" | grep "完成\|失敗" | while read -r line; do
            if [[ "$line" =~ \[([^]]+)\].*第${episode}集.*-\ ([^-]+)\ -\ ([^(]+).*耗時:\ ([^)]+) ]]; then
                local timestamp="${BASH_REMATCH[1]}"
                local step="${BASH_REMATCH[2]// /}"
                local status="${BASH_REMATCH[3]// /}"
                local duration="${BASH_REMATCH[4]}"
                
                local status_icon="✅"
                if [[ "$status" == "失敗" ]]; then
                    status_icon="❌"
                fi
                
                echo "  $status_icon $step: $duration ($timestamp)"
            fi
        done
    done
    
    echo ""
}

# 清理舊的日誌記錄
clean_timing_log() {
    local days_to_keep="${1:-7}"
    
    if [ ! -f "$TIMING_LOG_FILE" ]; then
        echo "❌ 時間統計檔案不存在"
        return 1
    fi
    
    # 備份原檔案
    cp "$TIMING_LOG_FILE" "${TIMING_LOG_FILE}.backup"
    
    # 計算截止日期
    local cutoff_date=$(date -d "$days_to_keep days ago" '+%Y-%m-%d')
    
    # 保留標題和最近的記錄
    head -4 "$TIMING_LOG_FILE" > "${TIMING_LOG_FILE}.tmp"
    grep -E "^\[" "$TIMING_LOG_FILE" | while read -r line; do
        if [[ "$line" =~ \[([0-9]{4}-[0-9]{2}-[0-9]{2}) ]]; then
            local log_date="${BASH_REMATCH[1]}"
            if [[ "$log_date" > "$cutoff_date" ]] || [[ "$log_date" == "$cutoff_date" ]]; then
                echo "$line" >> "${TIMING_LOG_FILE}.tmp"
            fi
        fi
    done
    
    mv "${TIMING_LOG_FILE}.tmp" "$TIMING_LOG_FILE"
    echo "✅ 已清理 $days_to_keep 天前的日誌記錄"
}
