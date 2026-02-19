.PHONY: proto install test

proto:
	python -m grpc_tools.protoc \
		-I proto \
		--python_out=generated \
		--grpc_python_out=generated \
		proto/recommender.proto
	# Fix protoc's bare relative import so the package import works correctly
	sed -i '' 's/import recommender_pb2 as recommender__pb2/from generated import recommender_pb2 as recommender__pb2/' \
		generated/recommender_pb2_grpc.py

install:
	pip install -r requirements.txt

test:
	pytest --cov=recommender tests/
