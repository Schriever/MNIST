package com.ml.ojezili.com.ml.ojeziliservice;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import javax.servlet.http.HttpServletResponse;

import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.slf4j.LoggerFactory;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;
import org.tensorflow.SavedModelBundle;


/*
 * Look into using Protocol buffers to implement this service in Python
 * 
 */

@RestController
@Component
public class ImagingRestService {
	private static final String DECODE_JPEG = "DecodeJpeg";
	private static final String DECODE_PNG = "DecodePng";
	private static final String PNG = "PNG";
	private static final String JPEG = "JPEG";
	private static final Logger LOGGER = LoggerFactory.getLogger(ImagingRestService.class);
	 	   
 
	private SavedModelBundle loadModelFromFile(String filename) {
		SavedModelBundle model = null;
		File modelfile=null;
		try {
			modelfile = new ClassPathResource(filename).getFile();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		String filePath = modelfile.toPath().toString();
		model = SavedModelBundle.load(filePath.substring(0, filePath.lastIndexOf("/")), "serve");
		
		return model;
	}

	
	private Tensor<Float> processImageIntoTensor(byte[] image, String fileType){
		Tensor<Float> imageTensor = null;
		Graph g = new Graph();
		
		Tensor<String> t = Tensor.<String>create(image, String.class);
	    Output<String> inputImage= g.opBuilder("Const", "input")
	    		.setAttr("dtype", DataType.fromClass(String.class))
	    		.setAttr("value", t)
	        	.build()
	        	.<String>output(0);
	        
        long chnls = 1;
	    String decodeType = null;
	    switch(fileType) {
	    case JPEG:	decodeType=DECODE_JPEG;
	    				break;
	    case PNG:	decodeType=DECODE_PNG;
	    				break;
	    	default:		LOGGER.error("Invalid image format");
	    				{g.close();return null;}
	    }
        Output<UInt8> decodedJpeg = g.opBuilder(decodeType, decodeType)
	    		.addInput(inputImage)
        		.setAttr("channels", chnls)
        		.build().<UInt8>output(0);     
	      
	    Output<Float> cast= g.opBuilder("Cast", "Cast")
	    		.addInput(decodedJpeg)
        		.setAttr("DstT", DataType.fromClass(Float.class))
        		.build()
        		.<Float>output(0);
	
	    Tensor<Integer> t3 = Tensor.<Integer>create(0, Integer.class);
	    Output<Integer> dimsConstant= g.opBuilder("Const", "make_batch")
	    		.setAttr("dtype", DataType.fromClass(Integer.class))
	    		.setAttr("value", t3)
	    		.build()
	    		.<Integer>output(0);
	      
	      
	    Output<Float> addDimension = g.opBuilder("ExpandDims", "ExpandDims").addInput(cast).addInput(dimsConstant).build().<Float>output(0);
	    
	    int[] sizearr= {28,28};
	  
	    Tensor<Integer> t1 = Tensor.<Integer>create(sizearr, Integer.class);
	    Output<Integer> sizeConstant= g.opBuilder("Const", "size")
	    		.setAttr("dtype", DataType.fromClass(Integer.class))
	    		.setAttr("value", t1)
	    		.build()
	    		.<Integer>output(0);
	     
	    Output<Float> resize = g.opBuilder("ResizeBilinear", "ResizeBilinear").addInput(addDimension).addInput(sizeConstant).build().<Float>output(0);
	     
	      
	      
	    Tensor<Integer> newshape = Tensor.<Integer>create(new int[] {1,784}, Integer.class);
	    Output<Integer> vectorShape = g.opBuilder("Const", "vector_shape")
	         .setAttr("dtype", newshape.dataType())
	         .setAttr("value", newshape)
	         .build()
             .<Integer>output(0);

	    Output<Float> reshaped= g.opBuilder("Reshape", "out")
	         .addInput(resize)
	         .addInput(vectorShape)
	         .build()
             .<Float>output(0);

		Session s = new Session(g);
		
		imageTensor = s.runner().fetch(reshaped.op().name()).run().get(0).expect(Float.class);
		s.close();
		g.close();
		return imageTensor;
	}
	 
	@RequestMapping (value= "/image/mnist", method= RequestMethod.POST)
	public String getMnistPrediction(@RequestParam("file") MultipartFile file, HttpServletResponse response) throws Exception{
		LOGGER.info("RestController getMnistPrediction => {filename} "+ file.getOriginalFilename());			
		byte[] imageFilebytes=null;
			
		if(validateFile(file)) {
			try {
				imageFilebytes = file.getBytes();
			} catch (Exception e) {
				LOGGER.error("Could not obtain file");
			}
		
		}
		
		SavedModelBundle model = loadModelFromFile("saved_model.pb"); 
		LOGGER.info("Loaded TF Model from file");
		
		String filetype = null;
		String ext = FilenameUtils.getExtension(file.getOriginalFilename());
		LOGGER.info("File extension: "+ext);
		switch(ext) {
		case "png":	filetype=PNG;
					break;
		case "jpeg": filetype=JPEG;
					break;
		case "jpg": filetype=JPEG;
					break;
		default	:	LOGGER.error("File Type not recognized");
					return "File Type not Recognized";
		}
		Tensor<Float> imageTensor = processImageIntoTensor(imageFilebytes, filetype);
		if(imageTensor==null) {
			LOGGER.error("Could not process Image Tensor");
			return "Error processing image";
		}
		else if(imageTensor!=null) {
		LOGGER.info("Processesd Image file into Tensor");
		}
		
	    long[]shape2= {10};
	    long[] output = new long[1];
	    float[] y = new float[10];
			
		Tensor<Float> out = Tensor.create(shape2, FloatBuffer.wrap(y));			
		Tensor<Long> result =model.session().runner()
				.feed("Placeholder",imageTensor )
				.feed("Placeholder_1", out)
				.fetch("ArgMax").run()
				.get(0)
				.expect(Long.class);
		result.copyTo(output);
		LOGGER.info("MNIST test predictated: "+output[0]);
		
		return "MNIST prediction: "+(output[0]);
	}
	
	@RequestMapping (value= "/image/cnnmnist", method= RequestMethod.POST)
	public String getCNNMnistPrediction(@RequestParam("file") MultipartFile file, HttpServletResponse response) throws Exception{
		LOGGER.info("RestController getMnistPrediction => {filename} "+ file.getOriginalFilename());			
		byte[] imageFilebytes=null;
			
		if(validateFile(file)) {
			try {
				imageFilebytes = file.getBytes();
			} catch (Exception e) {
				LOGGER.error("Could not obtain file");
			}
		
		}
		
		SavedModelBundle model = loadModelFromFile("convAWS2/saved_model.pb"); 
		LOGGER.info("Loaded TF Model from file");
		
		String filetype = null;
		String ext = FilenameUtils.getExtension(file.getOriginalFilename());
		LOGGER.info("File extension: "+ext);
		switch(ext) {
		case "png":	filetype=PNG;
					break;
		case "jpeg": filetype=JPEG;
					break;
		case "jpg": filetype=JPEG;
					break;
		default	:	LOGGER.error("File Type not recognized");
					return "File Type not Recognized";
		}
		Tensor<Float> imageTensor = processImageIntoTensor(imageFilebytes, filetype);
		if(imageTensor==null) {
			LOGGER.error("Could not process Image Tensor");
			return "Error processing image";
		}
		else if(imageTensor!=null) {
		LOGGER.info("Processesd Image file into Tensor");
		}
		
	    long[] output = new long[1];
	    long[] shape3= {1};
        float[] probability = {1};
			
	    Tensor<Float> prob = Tensor.create(shape3, FloatBuffer.wrap(probability));	
		Tensor<Long> result =model.session().runner()
				.feed("Placeholder_1",imageTensor )
				.feed("Placeholder_2", prob)
				.fetch("ArgMax").run()
				.get(0)
				.expect(Long.class);
		result.copyTo(output);
		LOGGER.info("CNN MNIST test predictated: "+output[0]);
		
		return "CNN MNIST prediction: "+(output[0]);
	}
	
	private boolean validateFile(MultipartFile file) {
		boolean result = true;
		if(file==null) result =false;
		if(file.isEmpty())result = false;
		if(file.getSize()<100)result = false;
		
		return result;
	}
}
