using UnityEngine;

public class YellowCarMovementZNegative : MonoBehaviour
{
    [SerializeField] private float speed;
    private float startX;
    private float startY;
    private Vector3 startLocation;   // Start position of the road
    private Vector3 endLocation = new Vector3(-318f, 0f, -300f);   // End position of the road

    private void Start()
    {
        // Store the initial X and Y values
        startX = transform.position.x;
        startY = transform.position.y;

        // Set the startLocation with initial X and Y values
        startLocation = new Vector3(startX, startY, 742f);
    }

    private void Update()
    {
        // Move the car back
        transform.Translate(Vector3.down * speed * Time.deltaTime);

        // Check if the car has reached the end of the road
        if (transform.position.z <= endLocation.z)
        {
            // Set the car at the beginning of the road
            transform.position = startLocation;
        }
    }
}