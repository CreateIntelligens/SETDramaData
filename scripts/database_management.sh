#!/bin/bash

# Database Management Functions
# 資料庫管理功能模組

# Load utility functions
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# Database management menu
show_database_menu() {
    while true; do
        echo ""
        echo "🗄️ 資料庫管理"
        echo "=============="
        echo "1. 查看資料庫統計 (Database Stats)"
        echo "2. 列出所有speaker (List All Speakers)"
        echo "3. 查看speaker詳細資訊 (Speaker Details)"
        echo "4. 查看集數speaker對應 (Episode Mapping)"
        echo "5. 匯出資料庫 (Export Database)"
        echo "6. 從JSON遷移 (Migrate from JSON)"
        echo "7. 備份資料庫 (Backup Database)"
        echo "8. 返回主選單 (Back to Main Menu)"
        echo ""
        echo -n "請選擇 [1-8]: "
        read choice
        
        case "$choice" in
            1)
                show_database_stats
                ;;
            2)
                list_all_speakers
                ;;
            3)
                show_speaker_details
                ;;
            4)
                show_episode_mapping
                ;;
            5)
                export_database
                ;;
            6)
                migrate_from_json_menu
                ;;
            7)
                backup_database
                ;;
            8)
                return
                ;;
            *)
                echo "❌ 無效選項"
                pause_for_input
                ;;
        esac
    done
}

# Database management functions
show_database_stats() {
    echo ""
    echo "📊 資料庫統計"
    echo "============"
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    $python_cmd "src/speaker_db_manager.py" stats
    pause_for_input
}

list_all_speakers() {
    echo ""
    echo "👥 所有Speaker"
    echo "============="
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    $python_cmd "src/speaker_db_manager.py" list
    pause_for_input
}

show_speaker_details() {
    echo ""
    echo "👤 Speaker詳細資訊"
    echo "=================="
    echo -n "請輸入Speaker ID: "
    read speaker_id
    
    if [ -z "$speaker_id" ]; then
        echo "❌ 請輸入有效的Speaker ID"
        pause_for_input
        return
    fi
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    echo ""
    $python_cmd "src/speaker_db_manager.py" speaker "$speaker_id"
    pause_for_input
}

show_episode_mapping() {
    echo ""
    echo "📺 集數Speaker對應"
    echo "=================="
    echo -n "請輸入集數: "
    read episode_num
    
    if [ -z "$episode_num" ]; then
        echo "❌ 請輸入有效的集數"
        pause_for_input
        return
    fi
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    echo ""
    $python_cmd "src/speaker_db_manager.py" episode "$episode_num"
    pause_for_input
}

export_database() {
    echo ""
    echo "📤 匯出資料庫"
    echo "============"
    echo -n "輸出檔案名稱 [預設: speakers_backup.json]: "
    read output_file
    output_file="${output_file:-speakers_backup.json}"
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    echo ""
    $python_cmd "src/speaker_db_manager.py" export "$output_file"
    pause_for_input
}

migrate_from_json_menu() {
    echo ""
    echo "🔄 從JSON遷移"
    echo "============="
    echo "此功能會將舊的processing_state.json轉換為SQLite資料庫"
    echo ""
    
    # Check if legacy JSON exists
    local json_file="processing_state.json"
    if [ ! -f "$json_file" ]; then
        echo "❌ 找不到JSON檔案: $json_file"
        pause_for_input
        return
    fi
    
    echo "找到JSON檔案: $json_file"
    if ! get_confirmation "確定要開始遷移嗎？"; then
        echo "❌ 已取消"
        pause_for_input
        return
    fi
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    echo ""
    $python_cmd "src/speaker_db_manager.py" migrate "$json_file"
    pause_for_input
}

backup_database() {
    echo ""
    echo "💾 備份資料庫"
    echo "============"
    
    if [ ! -f "speakers.db" ]; then
        echo "❌ 找不到資料庫檔案: speakers.db"
        pause_for_input
        return
    fi
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    $python_cmd "src/speaker_db_manager.py" backup
    pause_for_input
}