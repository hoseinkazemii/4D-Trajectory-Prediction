using UnityEngine;
using System;

public class DustCloudTrigger : MonoBehaviour
{
    public ParticleSystem dustCloudPrefab;
    private Transform grabbableObject;
    public GameObject craneHook;
    public static float taskEndTime;

    private float proximityThreshold = 0.5f; // Adjust this value to detect proximity
    public static bool taskEnded = false;

    public CSVWriter csvWriter;
    public VREyeTrackingDataManager vrEyeTracker; // Add reference to the VR eye tracker

    // Event to send collision data
    public static event Action<string, float, Vector3> OnCollisionLogged;

    void Start()
    {
        grabbableObject = transform;
    }

    void Update()
    {
        if (!grabbableObject.transform.IsChildOf(craneHook.transform))
        {
            // Check if grabbable object is close enough to the UnloadingStorey or UnloadingArea
            Collider[] colliders = Physics.OverlapSphere(grabbableObject.position, proximityThreshold);
            foreach (Collider collider in colliders)
            {
                if ((collider.gameObject.layer == LayerMask.NameToLayer("UnloadingStorey") || 
                    collider.gameObject.layer == LayerMask.NameToLayer("UnloadingArea")) && !taskEnded)
                {
                    RecordTaskEndTime();
                    taskEnded = true;
                    csvWriter.StopWriting(); // Stop writing the load movement (signals) data when the task ends
                    vrEyeTracker.StopEyeTracking(); // Stop writing VR eye tracking data when the task ends
                }
            }
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if ((collision.gameObject.layer == LayerMask.NameToLayer("UnloadingStorey") || collision.gameObject.layer == LayerMask.NameToLayer("UnloadingArea"))
            && !grabbableObject.transform.IsChildOf(craneHook.transform))
        {
            RecordTaskEndTime();
            // Instantiate dust cloud at the final position
            if (dustCloudPrefab != null)
            {
                ParticleSystem dustCloud = Instantiate(dustCloudPrefab, transform.position, Quaternion.identity);
                dustCloud.Play();
            }
        }
        // Log the collision event with name, time, and coordinates
        OnCollisionLogged?.Invoke(collision.gameObject.name, Time.time, grabbableObject.position);
    }

    private void RecordTaskEndTime()
    {
        if (!taskEnded)
        {
            taskEndTime = Time.time;
        }
    }
}



// using UnityEngine;
// using System;

// public class DustCloudTrigger : MonoBehaviour
// {
//     public ParticleSystem dustCloudPrefab;
//     private Transform grabbableObject;
//     public GameObject craneHook;
//     public static float taskEndTime;

//     private float proximityThreshold = 0.5f; // Adjust this value to detect proximity
//     public static bool taskEnded = false;

//     public CSVWriter csvWriter;
//     public VREyeTrackingDataManager vrEyeTracker; // Add reference to the VR eye tracker
//     public Metrics_Calculator metricsCalculator; // Reference to Metrics_Calculator instance

//     private float delayTimer; // The time to start proximity checks
//     private float startDelayTime = 15.0f; // Delay duration in seconds
//     private float pickupTime = -1f; // Initialize to an invalid time
//     private float gracePeriod = 1.0f; // Adjust this period based on the required delay

//     // Event to send collision data
//     public static event Action<string, float, Vector3> OnCollisionLogged;

//     void Start()
//     {
//         grabbableObject = transform;
//     }

//     public void SetDelayTimer(float startTime)
//     {
//         delayTimer = startTime; // Set delay timer from the main menu manager
//     }

//     void Update()
//     {
//         // Only proceed if the delay has passed
//         if (metricsCalculator.loadingStarted && Time.time >= delayTimer + startDelayTime)
//         {
//             // Check if the object is picked up and update pickupTime
//             if (grabbableObject.transform.IsChildOf(craneHook.transform) && pickupTime < 0)
//             {
//                 pickupTime = Time.time; // Record the time when the object is picked up
//             }

//             // Only check for proximity if the object is not a child or has passed the grace period
//             if (!grabbableObject.transform.IsChildOf(craneHook.transform) && Time.time >= pickupTime + gracePeriod)
//             {
//                 Collider[] colliders = Physics.OverlapSphere(grabbableObject.position, proximityThreshold);

//                 // Debug: Print the list of colliders detected within the proximity threshold
//                 // Debug.Log($"Colliders within proximity threshold at time: {Time.time:F2}");
//                 // foreach (Collider collider in colliders)
//                 // {
//                 //     Debug.Log($"Collider Name: {collider.gameObject.name}, Layer: {LayerMask.LayerToName(collider.gameObject.layer)}, Time: {Time.time:F2}");
//                 // }

//                 foreach (Collider collider in colliders)
//                 {
//                     if ((collider.gameObject.layer == LayerMask.NameToLayer("UnloadingStorey") || 
//                         collider.gameObject.layer == LayerMask.NameToLayer("UnloadingArea")) && !taskEnded)
//                     {
//                         // Debug.Log("HERE 1");
//                         EndTask();
//                     }
//                 }
//             }
//         }
//     }

//     private void OnCollisionEnter(Collision collision)
//     {
//         // Only proceed if the delay has passed
//         if (metricsCalculator.loadingStarted && Time.time >= delayTimer + startDelayTime &&
//             (collision.gameObject.layer == LayerMask.NameToLayer("UnloadingStorey") || 
//              collision.gameObject.layer == LayerMask.NameToLayer("UnloadingArea")) &&
//             !grabbableObject.transform.IsChildOf(craneHook.transform))
//         {
//             EndTask();

//             // Instantiate dust cloud at the final position
//             if (dustCloudPrefab != null)
//             {
//                 ParticleSystem dustCloud = Instantiate(dustCloudPrefab, transform.position, Quaternion.identity);
//                 dustCloud.Play();
//             }
//         }

//         // Log the collision event with name, time, and coordinates
//         OnCollisionLogged?.Invoke(collision.gameObject.name, Time.time, grabbableObject.position);
//     }

//     private void EndTask()
//     {
//         Debug.Log("Task end time: " + Time.time);
//         RecordTaskEndTime();
//         taskEnded = true;
//         csvWriter.StopWriting(); // Stop writing the load movement (signals) data when the task ends
//         vrEyeTracker.StopEyeTracking(); // Stop writing VR eye tracking data when the task ends
//     }

//     private void RecordTaskEndTime()
//     {
//         if (!taskEnded)
//         {
//             taskEndTime = Time.time;
//         }
//     }
// }





// using UnityEngine;

// public class DustCloudTrigger : MonoBehaviour
// {
//     public ParticleSystem dustCloudPrefab;
//     private Transform grabbableObject;
//     public GameObject craneHook;
//     public static float taskEndTime;

//     void Start()
//     {
//         grabbableObject = transform;
//     }

//     private void OnCollisionEnter(Collision collision)
//     {
//         if ((collision.gameObject.layer == LayerMask.NameToLayer("UnloadingStorey") || collision.gameObject.layer == LayerMask.NameToLayer("UnloadingArea"))
//             && !grabbableObject.transform.IsChildOf(craneHook.transform))
//         {
//             taskEndTime = Time.time;

//             if (dustCloudPrefab != null)
//             {
//                 ParticleSystem dustCloud = Instantiate(dustCloudPrefab, transform.position, Quaternion.identity);
//                 dustCloud.Play();
//             }
//         }
//     }
// }