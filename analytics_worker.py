import json
import time
import os
from collections import deque
from datetime import datetime
from confluent_kafka import Consumer, Producer
from analytics_utils import  process_data_for_metrics, calculate_metrics

# --- CONFIGURATION ---
# allow overriding via environment variables (useful in container)
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'localhost:9092')
TOPIC_RAW = os.environ.get('TOPIC_RAW', 'gaze_data')
TOPIC_AGG = os.environ.get('TOPIC_AGG', 'gaze_aggregates')
OUTPUT_FILE = os.environ.get('OUTPUT_FILE', 'dashboard_data.json')
LOG_FILE = os.environ.get('LOG_FILE', 'worker.log')

CONFIG = {
    'buffer_size': int(os.environ.get('BUFFER_SIZE', 300)),     
    'update_interval': float(os.environ.get('UPDATE_INTERVAL', 0.1)),
    'log_interval': float(os.environ.get('LOG_INTERVAL', 1.0)),
    'kafka_agg_interval': float(os.environ.get('KAFKA_AGG_INTERVAL', 5.0))
}

def log_to_file(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"[{ts}] {msg}\n")
    except:
        pass

def main():
    try:
        consumer = Consumer({
            'bootstrap.servers': KAFKA_BROKER,
            'group.id': 'analytics_english_v1', 
            'auto.offset.reset': 'latest'
        })
        consumer.subscribe([TOPIC_RAW])
        
        producer = Producer({'bootstrap.servers': KAFKA_BROKER})
        
        print(f"WORKER ANALYTICS ONLINE")
        print(f"Receiving envelopes from: {TOPIC_RAW}")
        
    except Exception as e:
        print(f"Kafka Error: {e}")
        return

    data_buffer = deque(maxlen=CONFIG['buffer_size'])
    table_rows = []
    
    last_dashboard_update = time.time()
    last_log_update = time.time()
    last_kafka_agg = time.time()

    try:
        while True:
            msg = consumer.poll(0.05)
            if msg is None: continue
            if msg.error(): continue

            try:
                # --- ENVELOPE UNPACKING ---
                raw_msg = json.loads(msg.value().decode('utf-8'))
                
                if 'metrics' in raw_msg:
                    packet = raw_msg['metrics']
                else:
                    packet = raw_msg # Legacy fallback
                
                data_buffer.append(packet)
                
                # --- NUOVO CALCOLO STRESS AVANZATO ---
                # Usiamo il buffer come finestra mobile per calcolare le metriche reali
                stress_val = 0
                if len(data_buffer) > 10:  # Calcoliamo solo se abbiamo abbastanza dati
                    # 1. Processiamo i dati nel buffer (aggiunge tempi e saccadi)
                    processed_buffer = process_data_for_metrics(list(data_buffer))
                    # 2. Calcoliamo le metriche aggregate sulla finestra mobile
                    current_metrics = calculate_metrics(processed_buffer)
                    # 3. Estraiamo lo stress score "serio"
                    stress_val = current_metrics['stress_score']
                
                current_time = time.time()
                current_blinks = packet.get('total_blinks', 0)
                is_talking = packet.get('is_talking', False)
                ear = packet.get('avg_ear', 0.0)

                # --- 1. LOG TABLE UPDATE ---
                if current_time - last_log_update >= CONFIG['log_interval']:
                    row = {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Stress": stress_val, # Ora è il valore reale!
                        "Blinks": current_blinks,
                        "Speaking": "YES" if is_talking else "NO",
                        "EAR": f"{ear:.3f}"
                    }
                    table_rows.insert(0, row)
                    if len(table_rows) > 15: table_rows.pop()
                    last_log_update = current_time

                # --- 2. KAFKA AGGREGATES ---
                if current_time - last_kafka_agg >= CONFIG['kafka_agg_interval']:
                    if len(data_buffer) > 0:
                        # Anche qui, usiamo lo stress_val calcolato seriamente
                        session_id = raw_msg.get('session_id', 'default_session')
                        
                        agg_payload = {
                            "timestamp": current_time,
                            "type": "PERIODIC_SUMMARY",
                            "session_id": session_id,
                            "metrics": {
                                "avg_stress": round(stress_val, 1), # Coerente con il dashboard
                                "total_blinks": current_blinks,
                                "is_talking": is_talking
                            }
                        }
                        producer.produce(TOPIC_AGG, 
                                       key=session_id.encode('utf-8'),
                                       value=json.dumps(agg_payload).encode('utf-8'))
                        producer.poll(0)
                        last_kafka_agg = current_time
                if current_time - last_dashboard_update >= CONFIG['update_interval']:
                    dashboard_payload = {
                        'metrics': {
                            'total_blinks': current_blinks,
                            'stress_score': stress_val,
                            'is_talking': is_talking,
                            'gaze_dir': f"{packet.get('gaze_horizontal','-')} - {packet.get('gaze_vertical','-')}"
                        },
                        'table_data': table_rows
                    }
                    
                    temp_file = OUTPUT_FILE + '.tmp'
                    with open(temp_file, 'w') as f: json.dump(dashboard_payload, f)
                    attempts = 0
                    while attempts < 5:
                        try: os.replace(temp_file, OUTPUT_FILE); break
                        except: time.sleep(0.01); attempts += 1
                    
                    last_dashboard_update = current_time

            except json.JSONDecodeError: continue
            except Exception: continue

    except KeyboardInterrupt: pass
    finally:
        consumer.close()
        producer.flush()

if __name__ == "__main__":
    main()