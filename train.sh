# The training process may collapse, use this to restart training automatically
for i in $(seq 1 100)
do
	python main.py
	sleep 5
done