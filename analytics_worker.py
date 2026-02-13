import json
import time
import os
from confluent_kafka import Consumer
from collections import deque
from datetime import datetime
from analytics_utils import (
    calculate_metrics,
    add_relative_time,
    calculate_saccade_distance,
    get_stress_level
)

# ========== CONFIGURAZIONE ==========
KAFKA_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'analytics_final_group', 
    'auto.offset.reset': 'latest',
    'enable.auto.commit': True,
    'auto.commit.interval.ms': 1000
}

TOPIC = 'gaze_data'
OUTPUT_FILE = 'dashboard_data.json'
STATS_FILE = 'worker_stats.json'  # NUOVO: File statistiche aggregate
LOG_FILE = 'worker.log'  # NUOVO: Log persistente

# NUOVO: Configurazione avanzata
CONFIG = {
    'buffer_size': 300,  # Frame da mantenere in memoria per calcoli
    'update_interval': 0.1,  # Secondi tra aggiornamenti dashboard
    'log_interval': 1.0,  # Secondi tra log nella tabella
    'stats_save_interval': 30,  # Secondi tra salvataggi statistiche
    'use_unified_stress': True,  # Usa calcolo stress da analytics_utils
}

# ========== NUOVE FUNZIONI ==========

def log_to_file(message, level="INFO"):
    """Scrive log su file con timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}\n"
    
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(log_entry)
    except:
        pass

def calculate_unified_stress(data_buffer):
    """
    Calcola stress usando analytics_utils per coerenza con l'app.
    Fallback al calcolo semplice se il buffer è troppo piccolo.
    """
    if len(data_buffer) < 10:
        # Fallback: calcolo semplice
        if data_buffer:
            last = data_buffer[-1]
            blinks = last.get('total_blinks', 0)
            ear = last.get('avg_ear', 0.3)
            return int(min(100, (blinks * 3) + (ear * 5)))
        return 0
    
    try:
        # Prepara dati per analytics_utils
        data_copy = list(data_buffer)
        data_with_time = add_relative_time(data_copy)
        data_with_saccades = calculate_saccade_distance(data_with_time)
        
        # Calcola metriche unificate
        metrics = calculate_metrics(data_with_saccades)
        return metrics['stress_score']
    
    except Exception as e:
        log_to_file(f"Errore calcolo stress unificato: {e}", "WARNING")
        # Fallback al calcolo semplice
        if data_buffer:
            last = data_buffer[-1]
            blinks = last.get('total_blinks', 0)
            ear = last.get('avg_ear', 0.3)
            return int(min(100, (blinks * 3) + (ear * 5)))
        return 0

def calculate_extended_metrics(data_buffer):
    """
    Calcola metriche estese dal buffer se abbastanza dati disponibili.
    """
    if len(data_buffer) < 20:
        return None
    
    try:
        data_copy = list(data_buffer)
        data_with_time = add_relative_time(data_copy)
        data_with_saccades = calculate_saccade_distance(data_with_time)
        metrics = calculate_metrics(data_with_saccades)
        return metrics
    except Exception as e:
        log_to_file(f"Errore calcolo metriche estese: {e}", "WARNING")
        return None

def save_stats(stats):
    """Salva statistiche aggregate su file"""
    try:
        temp_file = STATS_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(stats, f, indent=2)
        os.replace(temp_file, STATS_FILE)
    except Exception as e:
        log_to_file(f"Errore salvataggio stats: {e}", "ERROR")

def get_worker_health():
    """Restituisce informazioni sulla salute del worker"""
    return {
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': time.time() - start_time,
        'messages_processed': messages_processed,
        'errors_count': errors_count
    }

# ========== MAIN ==========

def main():
    global start_time, messages_processed, errors_count
    
    # NUOVO: Inizializzazione variabili globali
    start_time = time.time()
    messages_processed = 0
    errors_count = 0
    
    # NUOVO: Buffer dati per calcoli temporali
    data_buffer = deque(maxlen=CONFIG['buffer_size'])
    
    # NUOVO: Statistiche aggregate
    stats = {
        'total_sessions': 0,
        'total_blinks': 0,
        'max_stress_seen': 0,
        'total_talking_time': 0,
        'start_time': datetime.now().isoformat()
    }
    
    # Inizializzazione Kafka
    try:
        consumer = Consumer(KAFKA_CONFIG)
        consumer.subscribe([TOPIC])
        log_to_file("Worker Analytics avviato correttamente", "INFO")
        print(f"🧠 Worker Analytics Online. In ascolto per i blink...")
    except Exception as e:
        log_to_file(f"Errore inizializzazione Kafka: {e}", "CRITICAL")
        print(f"❌ Errore Kafka: {e}")
        return

    table_rows = []
    last_log_time = time.time()
    last_update_time = time.time()
    last_stats_save = time.time()

    try:
        while True:
            msg = consumer.poll(CONFIG['update_interval'])
            
            if msg is None: 
                continue
                
            if msg.error():
                errors_count += 1
                log_to_file(f"Errore Kafka: {msg.error()}", "ERROR")
                continue

            try:
                # 1. Ricevi il pacchetto dal Tracker
                packet = json.loads(msg.value().decode('utf-8'))
                messages_processed += 1
                
                # NUOVO: Aggiungi al buffer
                data_buffer.append(packet)
                
                # Prendi valori dal packet
                current_blinks = packet.get('total_blinks', 0)
                is_talking = packet.get('is_talking', False)
                avg_ear = packet.get('avg_ear', 0.0)
                
                # NUOVO: Aggiorna statistiche aggregate
                stats['total_blinks'] = max(stats['total_blinks'], current_blinks)
                if is_talking:
                    stats['total_talking_time'] += CONFIG['update_interval']

                # 2. CALCOLO STRESS (Unificato o Semplice)
                if CONFIG['use_unified_stress']:
                    stress_val = calculate_unified_stress(data_buffer)
                else:
                    # Calcolo semplice originale
                    stress_val = int(min(100, (current_blinks * 3) + (avg_ear * 5)))
                
                # NUOVO: Traccia stress massimo
                stats['max_stress_seen'] = max(stats['max_stress_seen'], stress_val)

                # 3. Aggiornamento Tabella Log (Con throttling)
                current_time = time.time()
                if current_time - last_log_time >= CONFIG['log_interval']:
                    row = {
                        "Orario": time.strftime("%H:%M:%S"),
                        "Stress": stress_val,
                        "Blink": current_blinks,
                        "Parlato": "SI" if is_talking else "NO",
                        "EAR": f"{avg_ear:.3f}"  # NUOVO: Aggiungi EAR
                    }
                    table_rows.insert(0, row)
                    if len(table_rows) > 20: 
                        table_rows.pop()
                    last_log_time = current_time

                # 4. NUOVO: Calcola metriche estese (se disponibili)
                extended_metrics = calculate_extended_metrics(data_buffer)

                # 5. Crea il pacchetto per la Dashboard Streamlit
                dashboard_payload = {
                    'metrics': {
                        'total_blinks': current_blinks,
                        'stress_score': stress_val,
                        'stress_level': get_stress_level(stress_val),  # NUOVO
                        'is_talking': is_talking,
                        'avg_ear': avg_ear,  # NUOVO
                        'gaze_dir': f"{packet['gaze_horizontal']} - {packet['gaze_vertical']}"
                    },
                    'table_data': table_rows,
                    'health': get_worker_health(),  # NUOVO
                    'buffer_size': len(data_buffer),  # NUOVO
                }
                
                # NUOVO: Aggiungi metriche estese se disponibili
                if extended_metrics:
                    dashboard_payload['extended_metrics'] = {
                        'blinks_per_minute': extended_metrics['blinks_per_minute'],
                        'gaze_variability': extended_metrics['gaze_variability'],
                        'scan_path_velocity': extended_metrics['scan_path_velocity'],
                        'pct_talking': extended_metrics['pct_talking'],
                        'fixation_count': extended_metrics['fixation_count']
                    }

                # 6. Scrittura su file JSON (Con throttling)
                if current_time - last_update_time >= CONFIG['update_interval']:
                    temp_file = OUTPUT_FILE + '.tmp'
                    with open(temp_file, 'w') as f:
                        json.dump(dashboard_payload, f)
                    os.replace(temp_file, OUTPUT_FILE)
                    last_update_time = current_time
                
                # NUOVO: Salvataggio periodico statistiche
                if current_time - last_stats_save >= CONFIG['stats_save_interval']:
                    save_stats(stats)
                    last_stats_save = current_time
                
                # Debug migliorato
                if messages_processed % 100 == 0:  # NUOVO: Log ogni 100 messaggi
                    debug_msg = (
                        f"[{messages_processed}] Blink: {current_blinks} | "
                        f"Stress: {stress_val} | Parlato: {is_talking} | "
                        f"Buffer: {len(data_buffer)}"
                    )
                    print(debug_msg)
                    log_to_file(debug_msg, "DEBUG")

            except json.JSONDecodeError as e:
                errors_count += 1
                log_to_file(f"Errore parsing JSON: {e}", "ERROR")
                continue
            except Exception as e:
                errors_count += 1
                log_to_file(f"Errore elaborazione messaggio: {e}", "ERROR")
                print(f"⚠️ Errore elaborazione: {e}")
                continue

    except KeyboardInterrupt:
        print("\n⏹️ Worker fermato manualmente.")
        log_to_file("Worker fermato manualmente", "INFO")
        
    except Exception as e:
        print(f"❌ Errore critico: {e}")
        log_to_file(f"Errore critico: {e}", "CRITICAL")
        
    finally:
        # NUOVO: Salvataggio finale
        print(f"\n📊 Statistiche finali:")
        print(f"  - Messaggi processati: {messages_processed}")
        print(f"  - Errori: {errors_count}")
        print(f"  - Uptime: {time.time() - start_time:.1f}s")
        print(f"  - Blink massimi: {stats['total_blinks']}")
        print(f"  - Stress massimo: {stats['max_stress_seen']}")
        
        # Salva stats finali
        stats['end_time'] = datetime.now().isoformat()
        stats['messages_processed'] = messages_processed
        stats['errors_count'] = errors_count
        save_stats(stats)
        log_to_file(f"Worker terminato. Messaggi: {messages_processed}, Errori: {errors_count}", "INFO")
        
        consumer.close()

if __name__ == "__main__":
    # NUOVO: Banner iniziale
    print("=" * 60)
    print("🧠 EYE TRACKING ANALYTICS WORKER v2.0")
    print("=" * 60)
    print(f"📝 Log file: {LOG_FILE}")
    print(f"📊 Stats file: {STATS_FILE}")
    print(f"💾 Output file: {OUTPUT_FILE}")
    print(f"🔧 Buffer size: {CONFIG['buffer_size']} frames")
    print(f"⚡ Unified stress: {'ENABLED' if CONFIG['use_unified_stress'] else 'DISABLED'}")
    print("=" * 60)
    print()
    
    main()