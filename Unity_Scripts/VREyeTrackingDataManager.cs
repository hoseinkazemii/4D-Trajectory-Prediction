using UnityEngine;
using System.IO;

public class VREyeTrackingDataManager : MonoBehaviour
{
    public static VREyeTrackingDataManager Instance;

    private string filePath;
    private StreamWriter csvWriter;

    private Vector3 leftGazeDirection;
    private Vector3 leftGazePosition;

    private Vector3 rightGazeDirection;
    private Vector3 rightGazePosition;

    private bool leftEyeUpdated = false;
    private bool rightEyeUpdated = false;

    private Vector3 previousObjectPosition = Vector3.zero;
    private string previousHitObjectName = "None";

    private bool isTaskActive = false; // Flag to track if the task has started

    // public GameObject validationDot; // Assign this in the Inspector

    public GameObject grabbableObject; // Assign this in the Inspector
    public GameObject craneHook; // Assign this in the Inspector (crane_hook with manually added box collider)

    private BoxCollider craneHookCollider; // To store the box collider of the crane hook

    void Start()
    {
        // Get the box collider of the crane hook
        craneHookCollider = craneHook.GetComponent<BoxCollider>();

        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Debug.LogWarning("Duplicate EyeTrackingDataManager found. Destroying this instance.");
            Destroy(gameObject);
        }
    }

    public void InitializeVREyeTracking(string scenarioName)
    {
        string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");
        filePath = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"VREyeTracking_{scenarioName}_{timestamp}.csv");

        csvWriter = new StreamWriter(filePath);
        csvWriter.WriteLine(
            "Timeframe,Gaze Direction X,Gaze Direction Y,Gaze Direction Z," +
            "Name,Gaze Position X,Gaze Position Y,Gaze Position Z," +
            "Object X,Object Y,Object Z,Shift Distance," +
            "Crane Hook X,Crane Hook Y,Crane Hook Z"
        );

        isTaskActive = true;  // Task is now active and the VR eye tracking data starts being exported
    }

    public void UpdateLeftEyeData(Vector3 gazeDirection, Vector3 gazePosition)
    {
        leftGazeDirection = gazeDirection;
        leftGazePosition = gazePosition;
        leftEyeUpdated = true;
        WriteDataIfReady();
    }

    public void UpdateRightEyeData(Vector3 gazeDirection, Vector3 gazePosition)
    {
        rightGazeDirection = gazeDirection;
        rightGazePosition = gazePosition;
        rightEyeUpdated = true;
        WriteDataIfReady();
    }
    
    private void WriteDataIfReady()
    {
        if (isTaskActive && leftEyeUpdated && rightEyeUpdated)
        {
            leftEyeUpdated = false;
            rightEyeUpdated = false;

            float currentTimeFrame = Time.time;

            // Average gaze direction and position
            Vector3 averageGazeDirection = (leftGazeDirection + rightGazeDirection) / 2;
            Vector3 averageGazePosition = (leftGazePosition + rightGazePosition) / 2;

            // Raycast to determine the hit object
            Ray ray = new Ray(averageGazePosition, averageGazeDirection);
            RaycastHit hit;
            string hitObjectName = "None";
            Vector3 hitObjectPosition = Vector3.zero;
            float shiftDistance = float.NaN;

            if (Physics.Raycast(ray, out hit, Mathf.Infinity))
            {
                // validationDot.transform.position = hit.point;

                hitObjectName = hit.collider.gameObject.name;
                // hitObjectPosition = hit.collider.gameObject.transform.position; // Gives the position of the parent object
                hitObjectPosition = hit.collider.bounds.center; // Gives the position of the box collider for the child objects

                // Calculate the shift distance if the object has changed
                if (previousHitObjectName != "None" && previousHitObjectName != hitObjectName)
                {
                    shiftDistance = Vector3.Distance(previousObjectPosition, hitObjectPosition);
                }

                // Update the previous object information
                previousHitObjectName = hitObjectName;
                previousObjectPosition = hitObjectPosition;
            }

            // Get crane hook position (from its box collider)
            Vector3 craneHookPosition = craneHookCollider.bounds.center;
            
            // Log data to CSV
            string logEntry = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}",
                                            currentTimeFrame,
                                            averageGazeDirection.x, averageGazeDirection.y, averageGazeDirection.z,
                                            hitObjectName,
                                            averageGazePosition.x, averageGazePosition.y, averageGazePosition.z,
                                            hitObjectPosition.x, hitObjectPosition.y, hitObjectPosition.z,
                                            shiftDistance,
                                            craneHookPosition.x, craneHookPosition.y, craneHookPosition.z
                                            );

            csvWriter.WriteLine(logEntry);
            csvWriter.Flush(); // Ensure data is written to the file in real time
        }
    }
    
    public void StopEyeTracking()
    {
        isTaskActive = false; // Stop writing data when the task ends
        if (csvWriter != null)
        {
            csvWriter.Close();
            csvWriter = null; // Release the writer object
        }
    }

    void OnApplicationQuit()
    {
        // Close the StreamWriter when the application quits
        if (csvWriter != null)
        {
            csvWriter.Close();
        }
    }
}








// using UnityEngine;
// using System.IO;

// public class VREyeTrackingDataManager : MonoBehaviour
// {
//     public static VREyeTrackingDataManager Instance;

//     private string filePath;
//     private StreamWriter csvWriter;
//     private float startTime;

//     private Vector3 leftGazeDirection;
//     private Vector3 leftGazePosition;

//     private Vector3 rightGazeDirection;
//     private Vector3 rightGazePosition;

//     private bool leftEyeUpdated = false;
//     private bool rightEyeUpdated = false;

//     private Vector3 previousObjectPosition = Vector3.zero;
//     private string previousHitObjectName = "None";

//     public GameObject validationDot; // Assign this in the Inspector
//     public GameObject powerline; // Assign this in the Inspector
//     private Collider powerlineCollider;

//     public GameObject grabbableObject; // Assign this in the Inspector
//     private Collider grabbableObjectCollider;

//     void Awake()
//     {
//         powerlineCollider = powerline.GetComponent<Collider>();
//         grabbableObjectCollider = grabbableObject.GetComponent<Collider>();

//         if (Instance == null)
//         {
//             Instance = this;
//             DontDestroyOnLoad(gameObject);

//             // Generate timestamp for the file name
//             string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");

//             // Set the file path to the Desktop with timestamp
//             filePath = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"VREyeTracking_{timestamp}.csv");

//             // Initialize StreamWriter
//             csvWriter = new StreamWriter(filePath);
//             csvWriter.WriteLine("Timeframe,Gaze Direction X,Gaze Direction Y,Gaze Direction Z,Name,Gaze Position X,Gaze Position Y,Gaze Position Z,Object X,Object Y,Object Z,Min Distance to Powerline (Hit Object),Min Distance to Powerline (Grabbable Object),Shift Distance");

//             // Record the start time
//             startTime = Time.time;
//         }
//         else
//         {
//             Debug.LogWarning("Duplicate EyeTrackingDataManager found. Destroying this instance.");
//             Destroy(gameObject);
//         }
//     }

//     public void UpdateLeftEyeData(Vector3 gazeDirection, Vector3 gazePosition)
//     {
//         leftGazeDirection = gazeDirection;
//         leftGazePosition = gazePosition;
//         leftEyeUpdated = true;
//         WriteDataIfReady();
//     }

//     public void UpdateRightEyeData(Vector3 gazeDirection, Vector3 gazePosition)
//     {
//         rightGazeDirection = gazeDirection;
//         rightGazePosition = gazePosition;
//         rightEyeUpdated = true;
//         WriteDataIfReady();
//     }

//     private void WriteDataIfReady()
//     {
//         if (leftEyeUpdated && rightEyeUpdated)
//         {
//             leftEyeUpdated = false;
//             rightEyeUpdated = false;

//             float currentTimeFrame = Time.time - startTime;

//             // Average gaze direction and position
//             Vector3 averageGazeDirection = (leftGazeDirection + rightGazeDirection) / 2;
//             Vector3 averageGazePosition = (leftGazePosition + rightGazePosition) / 2;

//             // Use the average gaze to determine the hit object
//             Ray ray = new Ray(averageGazePosition, averageGazeDirection);
//             RaycastHit hit;
//             string hitObjectName = "None";
//             float minDistanceToPowerlineHit = float.MaxValue;
//             Vector3 hitObjectPosition = Vector3.zero;
//             float shiftDistance = float.NaN;

//             if (Physics.Raycast(ray, out hit, Mathf.Infinity))
//             {
//                 validationDot.transform.position = hit.point;
//                 hitObjectName = hit.collider.gameObject.name;
//                 hitObjectPosition = hit.collider.gameObject.transform.position;

//                 // Calculate the minimum distance to the powerline for hit object
//                 minDistanceToPowerlineHit = Vector3.Distance(hit.point, powerlineCollider.ClosestPoint(hit.point));

//                 // Calculate the shift distance if the object has changed
//                 if (previousHitObjectName != "None" && previousHitObjectName != hitObjectName)
//                 {
//                     shiftDistance = Vector3.Distance(previousObjectPosition, hitObjectPosition);
//                 }

//                 // Update the previous object information
//                 previousHitObjectName = hitObjectName;
//                 previousObjectPosition = hitObjectPosition;
//             }

//             // Calculate the minimum distance from grabbable object to powerline
//             float minDistanceToPowerlineGrabbable = float.MaxValue;
//             if (grabbableObjectCollider != null && powerlineCollider != null)
//             {
//                 Vector3 grabbableObjectPosition = grabbableObject.transform.position;
//                 minDistanceToPowerlineGrabbable = Vector3.Distance(grabbableObjectPosition, powerlineCollider.ClosestPoint(grabbableObjectPosition));
//             }

//             // Log data to CSV in real time
//             string logEntry = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}",
//                                             currentTimeFrame,
//                                             averageGazeDirection.x, averageGazeDirection.y, averageGazeDirection.z,
//                                             hitObjectName,
//                                             averageGazePosition.x, averageGazePosition.y, averageGazePosition.z,
//                                             hitObjectPosition.x, hitObjectPosition.y, hitObjectPosition.z,
//                                             minDistanceToPowerlineHit,
//                                             minDistanceToPowerlineGrabbable,
//                                             shiftDistance);

//             csvWriter.WriteLine(logEntry);
//             csvWriter.Flush(); // Ensure data is written to the file in real time
//         }
//     }

//     void OnApplicationQuit()
//     {
//         // Close the StreamWriter when the application quits
//         if (csvWriter != null)
//         {
//             csvWriter.Close();
//         }
//     }
// }
