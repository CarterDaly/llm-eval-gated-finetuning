# app.py
from flask import Flask, request, jsonify
from celery import Celery
import time
import os

# Flask app setup
app = Flask(__name__)

# Celery configuration
app.config['broker_url'] = os.getenv('broker_url', 'redis://localhost:6379/0')
app.config['result_backend'] = os.getenv('result_backend', 'redis://localhost:6379/0')

# Create Celery instance
celery = Celery(app.name, broker=app.config['broker_url'])
celery.conf.update(app.config)

# Background task
@celery.task(bind=True)
def long_task(self, duration):
    """Simulates a long-running task."""
    for i in range(duration):
        time.sleep(1)
        self.update_state(state='PROGRESS', meta={'current': i + 1, 'total': duration})
    return {'status': 'Task completed!', 'total_seconds': duration}

@app.route('/start-task', methods=['POST'])
def start_task():
    """Start a background task."""
    try:
        print(request.data)
        duration = 5 #int(request.json.get('duration', 5))
        if duration <= 0:
            return jsonify({'error': 'Duration must be positive'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid duration'}), 400

    task = long_task.apply_async(args=[duration])
    return jsonify({'task_id': task.id}), 202

@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    """Check task status."""
    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({'state': task.state, 'status': 'Pending...'})
    elif task.state == 'PROGRESS':
        return jsonify({'state': task.state, 'progress': task.info})
    elif task.state == 'SUCCESS':
        return jsonify({'state': task.state, 'result': task.info})
    else:
        # FAILURE or RETRY
        return jsonify({'state': task.state, 'error': str(task.info)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# launch WSL
# Launch Redis
# python app.py
# celery -A app.celery worker --loglevel=info -P gevent
# curl.exe -X POST http://localhost:5000/start-task -H "Content-Type: application/json" -d '{"duration": "10"}'
# celery --broker=redis://localhost:6379/0 flower --address='localhost