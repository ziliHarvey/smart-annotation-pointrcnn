function Evaluation() {
	this.test_data = [];
	this.filenames = [];
	this.index = 0;

	this.evaluators = [];

	this.get_filename = function() {
		return this.filenames[this.index];
	}

	this.get_data = function() {
		return this.test_data[this.index];
	}

	this.next_frame = function() {
		this.index += 1;
	}

	this.is_done = function() {
		return this.index == this.filenames.length - 1;
	}

	this.add_data = function(data) {
		this.test_data.push(data);
	}

	this.add_filename = function(filename) {
		this.filenames.push(filename);
	}

	this.get_frame_number = function() {
		return this.index + 1;
	}

	this.num_frames = function() {
		return this.test_data.length;
	}

	this.add_evaluator = function(evaluator) {
		this.evaluators.push(evaluator);
	}
	
	// writes json output for current frame
	this.write_frame = function() {
		var output_evaluator = this.evaluators[this.evaluators.length-1].output();
		var output = {"frame": output_evaluator};
		var stringifiedOutput = JSON.stringify(output);
		// var file = new File([stringifiedOutput], this.get_filename().split(".")[0] + ".json", {type: "/json;charset=utf-8"});
		// saveAs(file);
		$.ajax({
            url: '/writeOutput',
            data: JSON.stringify({output: {filename: evaluator.get_filename(), 
                                            file: stringifiedOutput}}),
            type: 'POST',
            contentType: 'application/json;charset=UTF-8',
            success: function(response) {
                console.log("successfully saved output")
            },
            error: function(error) {
                console.log(error);
            }
        });
	}

	// writes json output for all frames
	this.write_output = function() {
		// output_bounding_boxes = save(this.)
		var output_evaluators = [];
		for (var i = 0; i < this.evaluators.length; i++) {
			// output_evaluators[this.filenames[i]] = this.evaluators[i].output();
			console.log(this.evaluators[i].output());
			output_evaluators.push(this.evaluators[i].output());
		}
		var output = {"frames": output_evaluators};
		var stringifiedOutput = JSON.stringify(output);
		var file = new File([stringifiedOutput], "output.json", {type: "/json;charset=utf-8"});
		saveAs(file);
	}

}

