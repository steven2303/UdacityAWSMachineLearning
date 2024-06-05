import json

THRESHOLD = .93

def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = json.loads(event['inferences'])
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(inference >= THRESHOLD for inference in inferences)
    
    # If our threshold is met, pass our data back out of the Step Function
    if meets_threshold:
        return {
            'statusCode': 200,
            'body': json.dumps(event)
        }
    else:
        # End the Step Function with an error
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
