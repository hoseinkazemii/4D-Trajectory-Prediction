using UnityEngine;

public class EyeTrackingDataCaptureRight : MonoBehaviour
{
    private OVREyeGaze eyeGaze;
    public VREyeTrackingDataManager vrEyeTrackingDataManager; // Assign this in the Inspector

    void Start()
    {
        eyeGaze = GetComponent<OVREyeGaze>();
        
        // Start the repeating method call every 0.02 seconds
        InvokeRepeating("CaptureEyeTrackingData", 0f, 0.02f);
    }

    void CaptureEyeTrackingData()
    {
        if (eyeGaze != null && OVRManager.eyeFovPremultipliedAlphaModeEnabled)
        {
            // Get gaze direction and position
            Vector3 gazeDirection = eyeGaze.transform.forward;
            Vector3 gazePosition = eyeGaze.transform.position;

            // Update the data manager
            vrEyeTrackingDataManager.UpdateRightEyeData(gazeDirection, gazePosition);
        }
    }
}




// using UnityEngine;
// using System.IO;

// public class EyeTrackingDataCaptureRight : MonoBehaviour
// {
//     private OVREyeGaze eyeGaze;
//     public GameObject validationDot;

//     private string filePath;
//     private StreamWriter csvWriterEyeTrackingVR;
//     private float startTime;

//     void Start()
//     {
//         eyeGaze = GetComponent<OVREyeGaze>();

//         // Generate timestamp for the file name
//         string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");

//         // Set the file path to the Desktop with timestamp
//         filePath = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"EyeTrackingData_{timestamp}.csv");

//         // Initialize StreamWriter
//         csvWriterEyeTrackingVR = new StreamWriter(filePath);
//         csvWriterEyeTrackingVR.WriteLine("Timeframe,Gaze Direction X,Gaze Direction Y,Gaze Direction Z,Name,Gaze Position X,Gaze Position Y,Gaze Position Z");

//         // Record the start time
//         startTime = Time.time;
//     }

//     void Update()
//     {
//         if (eyeGaze != null && OVRManager.eyeFovPremultipliedAlphaModeEnabled)
//         {
//             Ray ray = new Ray(eyeGaze.transform.position, eyeGaze.transform.forward);
//             RaycastHit hit;

//             string hitObjectName = "None";

//             if (Physics.Raycast(ray, out hit, Mathf.Infinity))
//             {
//                 validationDot.transform.position = hit.point;
//                 hitObjectName = hit.collider.gameObject.name;
//             }

//             float currentTimeFrame = Time.time - startTime;

//             // Get gaze direction and position
//             Vector3 gazeDirection = eyeGaze.transform.forward;
//             Vector3 gazePosition = eyeGaze.transform.position;

//             Debug.Log("gazeDirection Right: " + gazeDirection);

//             // Log data to CSV in real time
//             string logEntry = string.Format("{0},{1},{2},{3},{4},{5},{6},{7}",
//                                             currentTimeFrame,
//                                             gazeDirection.x, gazeDirection.y, gazeDirection.z,
//                                             hitObjectName,
//                                             gazePosition.x, gazePosition.y, gazePosition.z);

//             csvWriterEyeTrackingVR.WriteLine(logEntry);
//             csvWriterEyeTrackingVR.Flush(); // Ensure data is written to the file in real time
//         }
//     }

//     void OnApplicationQuit()
//     {
//         // Close the StreamWriter when the application quits
//         if (csvWriterEyeTrackingVR != null)
//         {
//             csvWriterEyeTrackingVR.Close();
//         }
//     }
// }
