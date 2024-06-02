for i in data/parkinsons_data/*.fbx; do
	python custom_retarget_motion.py --visualize False --filename $i
done

