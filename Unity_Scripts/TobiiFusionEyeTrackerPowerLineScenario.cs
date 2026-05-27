using UnityEngine;
using Tobii.Research.Unity;
using System.IO;

public class PCEyeTrackerPowerline : MonoBehaviour
{
    private EyeTracker eyeTracker;

    private bool isWriting = true; // Flag to control writing
    private string filePath;
    private StreamWriter eyeTrackingCsvWriter;

    public GameObject powerline; // Assign this in the Inspector
    private Collider powerlineCollider;

    public GameObject grabbableObject; // Assign this in the Inspector
    public GameObject craneHook; // Assign this in the Inspector (the object with box collider for crane_hook)

    private BoxCollider craneHookCollider; // To store the box collider of the crane hook

    float minDistanceToPowerlineGrabbable = float.MaxValue;
    private float xDistancePowerlineGrabbable;
    private float yDistancePowerlineGrabbable;
    private float zDistancePowerlineGrabbable;

    private Vector3 previousObjectPosition = Vector3.zero;
    private string previousHitObjectName = "None";

    void Start()
    {
        eyeTracker = EyeTracker.Instance;
        powerlineCollider = powerline.GetComponent<Collider>();

        // Get the box collider of the crane hook
        craneHookCollider = craneHook.GetComponent<BoxCollider>();
    }

    public void InitializePCEyeTracker(string scenarioName)
    {
        // Generate timestamp for the file name
        string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");

        // Set the file path to the Desktop with timestamp
        filePath = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"TobiiEyeTracking_{scenarioName}_{timestamp}.csv");

        // Initialize StreamWriter
        eyeTrackingCsvWriter = new StreamWriter(filePath);
        eyeTrackingCsvWriter.WriteLine(
            "Timeframe,Gaze Direction X,Gaze Direction Y,Gaze Direction Z," +
            "Name,Gaze Position X,Gaze Position Y,Gaze Position Z," +
            "Object X,Object Y,Object Z,Min Distance to Powerline (Hit Object)," +
            "Min Distance to Powerline (Grabbable Object),Shift Distance," +
            "Powerline and Grabbable X Distance,Powerline and Grabbable Y Distance," +
            "Powerline and Grabbable Z Distance,Crane Hook X,Crane Hook Y,Crane Hook Z," +
            "Right Pupil Diameter,Left Pupil Diameter"
        );
        
        // Start the repeating method call every 0.02 seconds
        InvokeRepeating("CaptureEyeTrackingData", 0f, 0.02f);
    }

    void CaptureEyeTrackingData()
    {
        IGazeData gazeData = eyeTracker.LatestGazeData;

        if (gazeData != null && gazeData.CombinedGazeRayScreenValid)
        {
            Ray gazeRay = gazeData.CombinedGazeRayScreen;
            RaycastHit hit;

            string hitObjectName = "None";
            Vector3 hitObjectPosition = Vector3.zero;
            float minDistanceToPowerlineHit = float.MaxValue;
            float shiftDistance = float.NaN;

            // Perform raycast to find the hit object
            if (Physics.Raycast(gazeRay, out hit))
            {
                hitObjectName = hit.collider.gameObject.name;
                // hitObjectPosition = hit.collider.gameObject.transform.position; // Gives the position of the parent object
                hitObjectPosition = hit.collider.bounds.center; // Gives the position of the box collider for the child objects

                // Calculate the minimum distance to the powerline for the hit object
                minDistanceToPowerlineHit = Vector3.Distance(hit.point, powerlineCollider.ClosestPoint(hit.point));

                // Calculate the shift distance if the object has changed
                if (previousHitObjectName != "None" && previousHitObjectName != hitObjectName)
                {
                    shiftDistance = Vector3.Distance(previousObjectPosition, hitObjectPosition);
                }

                // Update previous object information
                previousHitObjectName = hitObjectName;
                previousObjectPosition = hitObjectPosition;
            }

            float currentTime = Time.time;
            Vector3 gazeDirection = gazeRay.direction;
            Vector3 gazePosition = gazeRay.origin;

            // Calculate the minimum distance from the grabbable object to the powerline
            Vector3 grabbableObjectPosition = grabbableObject.transform.position;
            minDistanceToPowerlineGrabbable = Vector3.Distance(grabbableObjectPosition, powerlineCollider.ClosestPoint(grabbableObjectPosition));
            xDistancePowerlineGrabbable = Mathf.Abs(grabbableObjectPosition.x - powerline.transform.position.x);
            yDistancePowerlineGrabbable = Mathf.Abs(grabbableObjectPosition.y - powerline.transform.position.y);
            zDistancePowerlineGrabbable = Mathf.Abs(grabbableObjectPosition.z - powerline.transform.position.z);

            // Get crane hook position (from its box collider)
            Vector3 craneHookPosition = craneHookCollider.bounds.center;

            // Get pupil diameters
            float rightPupilDiameter = gazeData.Right.PupilDiameter;
            float leftPupilDiameter = gazeData.Left.PupilDiameter;

            // Log data to CSV in real time
            string logEntry = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21}",
                                            currentTime,
                                            gazeDirection.x, gazeDirection.y, gazeDirection.z,
                                            hitObjectName,
                                            gazePosition.x, gazePosition.y, gazePosition.z,
                                            hitObjectPosition.x, hitObjectPosition.y, hitObjectPosition.z,
                                            minDistanceToPowerlineHit,
                                            minDistanceToPowerlineGrabbable,
                                            shiftDistance,
                                            xDistancePowerlineGrabbable,
                                            yDistancePowerlineGrabbable,
                                            zDistancePowerlineGrabbable,
                                            craneHookPosition.x, craneHookPosition.y, craneHookPosition.z,
                                            rightPupilDiameter,
                                            leftPupilDiameter);

            if (isWriting)
            {
                eyeTrackingCsvWriter.WriteLine(logEntry);
                eyeTrackingCsvWriter.Flush(); // Ensure data is written to the file in real time
            }
        }
    }

    // Method to stop writing when the task ends
    public void StopWriting()
    {
        isWriting = false; // Set the flag to stop writing
        if (eyeTrackingCsvWriter != null)
        {
            eyeTrackingCsvWriter.Close();
            eyeTrackingCsvWriter = null; // Release the writer object
        }
    }

    void OnApplicationQuit()
    {
        // Close the StreamWriter when the application quits
        if (eyeTrackingCsvWriter != null)
        {
            eyeTrackingCsvWriter.Close();
        }
    }
}



//// Tobii eye-tracking before adding unifying VR and PC eye tracking data exportation
// using UnityEngine;
// using Tobii.Research.Unity;
// using System.IO;

// public class PCEyeTracker : MonoBehaviour
// {
//     private EyeTracker eyeTracker;

//     private bool isWriting = true; // Flag to control writing
//     private string filePath;
//     private StreamWriter eyeTrackingCsvWriter;
//     private float startTime;

//     void Start()
//     {
//         eyeTracker = EyeTracker.Instance;

//         // Record the start time
//         startTime = Time.time;

//         // Start the repeating method call every 0.02 seconds
//         InvokeRepeating("CaptureEyeTrackingData", 0f, 0.02f);
//     }

//     public void InitializePCEyeTracker(string scenarioName)
//     {
//         // Generate timestamp for the file name
//         string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");

//         // Set the file path to the Desktop with timestamp
//         filePath = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"TobiiEyeTracking_{scenarioName}_{timestamp}.csv");

//         // Initialize StreamWriter
//         eyeTrackingCsvWriter = new StreamWriter(filePath);
//         eyeTrackingCsvWriter.WriteLine("Time,Name,Right Pupil Diameter,Left Pupil Diameter");
//     }

//     void CaptureEyeTrackingData()
//     {
//         IGazeData gazeData = eyeTracker.LatestGazeData;

//         if (gazeData != null && gazeData.CombinedGazeRayScreenValid)
//         {
//             Ray gazeRay = gazeData.CombinedGazeRayScreen;
//             RaycastHit hit;

//             string hitObjectName = "None";

//             if (Physics.Raycast(gazeRay, out hit))
//             {
//                 hitObjectName = hit.collider.gameObject.name;
//                 Debug.Log("Looking at: " + hitObjectName);
//                 Debug.DrawRay(gazeRay.origin, gazeRay.direction * hit.distance, Color.yellow);
//             }

//             float currentTime = Time.time - startTime;
//             float rightPupilDiameter = gazeData.Right.PupilDiameter;
//             float leftPupilDiameter = gazeData.Left.PupilDiameter;

//             // Log data to CSV in real time
//             string logEntry = string.Format("{0},{1},{2},{3}",
//                                             currentTime,
//                                             hitObjectName,
//                                             rightPupilDiameter,
//                                             leftPupilDiameter);
//             if (isWriting)
//             {
//             eyeTrackingCsvWriter.WriteLine(logEntry);
//             eyeTrackingCsvWriter.Flush(); // Ensure data is written to the file in real time
//             }
//         }
//     }

//     // Method to stop writing when the task ends
//     public void StopWriting()
//     {
//         isWriting = false; // Set the flag to stop writing
//         if (eyeTrackingCsvWriter != null)
//         {
//             eyeTrackingCsvWriter.Close();
//             eyeTrackingCsvWriter = null; // Release the writer object
//         }
//     }

//     void OnApplicationQuit()
//     {
//         // Close the StreamWriter when the application quits
//         if (eyeTrackingCsvWriter != null)
//         {
//             eyeTrackingCsvWriter.Close();
//         }
//     }
// }
