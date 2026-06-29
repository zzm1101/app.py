# check_db.py
import sqlite3
import json

# 检查原系统数据库
print("=" * 60)
print("检查原系统数据库 (yogurt_qlf.db)")
print("=" * 60)

try:
    conn = sqlite3.connect('yogurt_qlf.db')
    cursor = conn.cursor()
    
    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_settings'")
    if cursor.fetchone():
        cursor.execute("SELECT key, value FROM system_settings")
        rows = cursor.fetchall()
        if rows:
            print("\n📦 system_settings 表内容:")
            for key, value in rows:
                print(f"\n  键: {key}")
                # 尝试解析 JSON
                try:
                    data = json.loads(value)
                    print(f"  值类型: JSON 对象")
                    print(f"  内容预览: {json.dumps(data, ensure_ascii=False)[:200]}...")
                except:
                    print(f"  值: {value[:200]}...")
        else:
            print("\n⚠️ system_settings 表为空")
    else:
        print("\n⚠️ system_settings 表不存在")
    
    conn.close()
except Exception as e:
    print(f"错误: {e}")

# 检查密封系统数据库
print("\n" + "=" * 60)
print("检查密封系统数据库 (seal_history.db)")
print("=" * 60)

try:
    conn = sqlite3.connect('seal_history.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_settings'")
    if cursor.fetchone():
        cursor.execute("SELECT key, value FROM system_settings")
        rows = cursor.fetchall()
        if rows:
            print("\n📦 system_settings 表内容:")
            for key, value in rows:
                print(f"\n  键: {key}")
                try:
                    data = json.loads(value)
                    print(f"  值类型: JSON 对象")
                    print(f"  内容预览: {json.dumps(data, ensure_ascii=False)[:200]}...")
                except:
                    print(f"  值: {value[:200]}...")
        else:
            print("\n⚠️ system_settings 表为空")
    else:
        print("\n⚠️ system_settings 表不存在")
    
    conn.close()
except Exception as e:
    print(f"错误: {e}")

print("\n" + "=" * 60)
print("检查完成")
input("按回车键退出...")